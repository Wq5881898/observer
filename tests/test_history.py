import sqlite3,tempfile,unittest
from contextlib import closing
from pathlib import Path
from datetime import date,datetime,timedelta,timezone
from zoneinfo import ZoneInfo
import numpy as np
from observer.history import month_window,minutes_text,heat_level
from observer.product import ProductStore,SessionService,SCHEMA_VERSION

class HistoryTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.root=Path(self.temp.name)
        self.store=ProductStore(self.root);self.service=SessionService(self.store)
        self.now=datetime(2026,11,15,12,tzinfo=timezone.utc)
    def tearDown(self):
        self.store.close();self.temp.cleanup()
    def session(self,start,seconds,source='test',title='Task'):
        sid=self.service.start(title,1440,0,start,{'source':source})
        self.service.end(seconds,start+timedelta(seconds=seconds));return sid
    def calendar(self,now=None,zone='UTC'):
        return self.store.history_calendar(now or self.now,zone)
    def test_natural_month_window_crosses_year_and_leap_month(self):
        self.assertEqual(month_window(date(2027,1,15)),(date(2026,8,1),date(2027,2,1)))
        self.assertEqual(month_window(date(2028,2,29)),(date(2027,9,1),date(2028,3,1)))

    def test_half_year_includes_sixth_month_without_deleting_older_history(self):
        included=self.session(datetime(2026,6,1,12,tzinfo=timezone.utc),600)
        older=self.session(datetime(2026,5,31,12,tzinfo=timezone.utc),600)
        result=self.calendar()
        self.assertEqual(result['start'],'2026-06-01')
        self.assertEqual({r['id'] for r in result['rows']},{included})
        self.assertIn(older,{r['id'] for r in self.store.history()})
    def test_cross_midnight_pauses_and_breaks_count_only_work(self):
        start=datetime(2026,11,14,23,50,tzinfo=timezone.utc)
        sid=self.service.start('Split',180,0,start,{'source':'test'},break_every_minutes=10,break_minutes=5)
        self.service.pause(600,start+timedelta(seconds=600))
        self.service.resume(1200,start+timedelta(seconds=1200))
        self.service.start_break(1800,start+timedelta(seconds=1800))
        self.service.finish_break(2100,start+timedelta(seconds=2100))
        self.service.end(2400,start+timedelta(seconds=2400))
        result=self.calendar()
        self.assertEqual(result['days']['2026-11-14']['seconds'],600)
        self.assertEqual(result['days']['2026-11-15']['seconds'],900)
        self.assertEqual({r['id'] for r in result['rows']},{sid})
        self.assertEqual(sum(r['seconds'] for r in result['rows']),1500)
        self.assertEqual(self.store.summary(sid)['scoring_seconds'],1500)
    def test_demo_does_not_inflate_heatmap(self):
        self.session(self.now-timedelta(hours=2),600)
        self.session(self.now-timedelta(hours=1),3600,'synthetic-demo')
        result=self.calendar();info=result['days']['2026-11-15']
        self.assertEqual(info['seconds'],600);self.assertEqual(info['tasks'],1)
        self.assertEqual(info['demo_tasks'],1);self.assertEqual(len(result['rows']),2)
    def test_old_cross_day_record_is_not_fabricated(self):
        sid=self.session(datetime(2026,11,14,23,50,tzinfo=timezone.utc),1200)
        with self.store.db:self.store.db.execute('DELETE FROM session_work WHERE session_id=?',(sid,))
        result=self.calendar();self.assertEqual(result['days']['2026-11-14']['seconds'],1200)
        self.assertNotIn('2026-11-15',result['days'])
        self.assertIn('旧记录',result['rows'][0]['timing_note'])
    def test_missing_legacy_duration_is_not_planned_duration(self):
        sid=self.session(self.now-timedelta(hours=1),600)
        with self.store.db:
            self.store.db.execute('DELETE FROM session_work WHERE session_id=?',(sid,))
            self.store.db.execute('DELETE FROM totals WHERE session_id=?',(sid,))
        row=self.calendar()['rows'][0];self.assertIsNone(row['seconds'])
        self.assertEqual(minutes_text(row['seconds']),'时长未知')
    def test_calendar_and_history_do_not_truncate_at_100(self):
        for i in range(121):self.session(self.now-timedelta(hours=4)+timedelta(minutes=i),60)
        self.assertEqual(len(self.store.history()),121)
        result=self.calendar();self.assertEqual(len(result['rows']),121)
        self.assertEqual(result['days']['2026-11-15']['seconds'],121*60)
    def test_v9_migration_preserves_records_and_creates_backup(self):
        sid=self.session(self.now-timedelta(hours=1),60);self.store.close()
        with closing(sqlite3.connect(self.root/'observer.sqlite3')) as db:
            with db:
                db.execute('DROP TABLE session_work');db.execute('DELETE FROM schema_migrations WHERE version=10');db.execute('PRAGMA user_version=9')
        self.store=ProductStore(self.root)
        self.assertEqual(self.store.db.execute('PRAGMA user_version').fetchone()[0],SCHEMA_VERSION)
        self.assertEqual(self.store.summary(sid)['scoring_seconds'],60)
        self.assertEqual(self.calendar()['rows'][0]['seconds'],60)
        with closing(sqlite3.connect(self.root/'observer.sqlite3.v9.backup')) as db:
            self.assertEqual(db.execute('PRAGMA user_version').fetchone()[0],9)
            self.assertEqual(db.execute('SELECT count(*) FROM sessions').fetchone()[0],1)
    def test_dst_repeated_hour_counts_elapsed_minutes(self):
        start=datetime(2026,11,1,0,30,tzinfo=ZoneInfo('America/Edmonton')).astimezone(timezone.utc)
        self.session(start,3*3600)
        result=self.calendar(zone='America/Edmonton')
        self.assertEqual(result['days']['2026-11-01']['seconds'],10800)
    def test_midnight_end_does_not_add_empty_next_day(self):
        self.session(datetime(2026,11,14,23,50,tzinfo=timezone.utc),600)
        self.assertNotIn('2026-11-15',self.calendar()['days'])
    def test_repeated_wall_start_does_not_overwrite_previous_work(self):
        start=self.now-timedelta(hours=1)
        self.service.start('Clock',120,0,start,{'source':'test'})
        self.service.pause(600,start+timedelta(seconds=600))
        self.service.resume(1200,start)
        self.service.end(1800,start+timedelta(seconds=600))
        self.assertEqual(self.calendar()['days']['2026-11-15']['seconds'],1200)
    def test_expired_screenshot_preserves_logs_and_is_not_an_error(self):
        self.store.set_images(True);start=self.now-timedelta(days=31)
        sid=self.service.start('Keep text',1,0,start,{'source':'test'})
        self.assertTrue(self.store.capture(sid,'picture',np.zeros((16,16,3),np.uint8)))
        self.service.end(60,start+timedelta(seconds=60))
        before=self.store.summary(sid)
        self.assertEqual(self.store.purge(self.now),1)
        after=self.store.summary(sid)
        self.assertEqual(before['actions'],after['actions'])
        self.assertEqual(after['scoring_seconds'],60)
        self.assertIsNotNone(after['evidence'][0]['expired_at'])
        self.assertIsNone(after['evidence'][0]['error'])
        self.assertEqual(self.store.data_status()['evidence_errors'],0)
        self.assertEqual(self.store.purge(self.now),0)
        self.store.delete_session(sid);self.assertEqual(self.store.history(),[])
    def test_capture_date_not_long_pause_end_controls_expiry(self):
        self.store.set_images(True);start=self.now-timedelta(days=31)
        sid=self.service.start('Long pause',120,0,start,{'source':'test'})
        self.store.capture(sid,'old',np.zeros((16,16,3),np.uint8),offset=0)
        self.service.pause(60,start+timedelta(seconds=60))
        at=31*86400-60;self.service.resume(at,self.now-timedelta(seconds=60))
        self.store.capture(sid,'recent',np.zeros((16,16,3),np.uint8),offset=at)
        self.service.end(at+60,self.now)
        self.assertEqual(self.store.purge(self.now),1)
        expired={r['id']:bool(r['expired_at']) for r in self.store.summary(sid)['evidence']}
        self.assertEqual(expired,{'old':True,'recent':False})

    def test_expiry_keeps_original_capture_error(self):
        self.store.set_images(True);start=self.now-timedelta(days=31)
        sid=self.service.start('Keep error',1,0,start,{'source':'test'})
        self.store.capture(sid,'failed',None)
        self.service.end(60,start+timedelta(seconds=60))
        error=self.store.summary(sid)['evidence'][0]['error']
        self.assertTrue(error)
        self.store.purge(self.now)
        self.assertEqual(self.store.summary(sid)['evidence'][0]['error'],error)
    def test_delete_removes_daily_total_and_intervals(self):
        sid=self.session(self.now-timedelta(hours=1),600);self.store.delete_session(sid)
        self.assertEqual(self.calendar()['days'],{})
        self.assertEqual(self.store.db.execute('SELECT count(*) FROM session_work').fetchone()[0],0)
    def test_format_and_color_boundaries(self):
        self.assertEqual(minutes_text(3600),'60分钟');self.assertEqual(minutes_text(90),'1.5分钟')
        self.assertEqual([heat_level(s*60) for s in [0,1,29,30,60,120,240]],[0,1,1,2,3,4,5])
