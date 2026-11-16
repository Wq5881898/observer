import importlib.util
import os
from pathlib import Path
import tempfile
import time
from datetime import datetime,timedelta,timezone,date
from types import SimpleNamespace
import unittest
from zoneinfo import ZoneInfo

os.environ.setdefault('QT_QPA_PLATFORM','offscreen')
HAS_QT=importlib.util.find_spec('PySide6') is not None
if HAS_QT:
    from PySide6.QtWidgets import QApplication
    from observer.desktop import ObserverWindow
    from observer.history_calendar import HistoryCalendar,COLORS
from observer.product import ProductStore,SessionService
from observer.history import month_window
from observer.local_timezone import get_localzone_name

@unittest.skipUnless(HAS_QT,'desktop optional dependency not installed')
class HistoryPageTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app=QApplication.instance() or QApplication([])
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.root=Path(self.temp.name)
        self.window=None
        self.zone=ZoneInfo(get_localzone_name())
        self.today=datetime.now(self.zone).date()
    def tearDown(self):
        if self.window:
            self.window.request_exit();self.wait_for(lambda:not self.window.worker.isRunning(),10)
            self.window.close();self.app.processEvents()
        self.temp.cleanup()
    def wait_for(self,check,seconds=6):
        end=time.monotonic()+seconds
        while not check() and time.monotonic()<end:self.app.processEvents();time.sleep(.01)
        self.app.processEvents();self.assertTrue(check())
    def seed(self,day,seconds,title):
        start=datetime.combine(day,datetime.min.time(),self.zone).astimezone(timezone.utc)
        store=ProductStore(self.root)
        try:
            service=SessionService(store);sid=service.start(title,1440,0,start,{'source':'test'})
            service.end(seconds,start+timedelta(seconds=seconds));return sid
        finally:store.close()
    def open(self):
        self.window=ObserverWindow(SimpleNamespace(data_dir=self.root,demo=True,models_dir=Path('models'),profile=None))
        self.window.show();self.wait_for(lambda:self.window.start.isEnabled())
        self.wait_for(lambda:self.window.history_payload is not None)
    def test_default_today_and_approved_duration_before_title(self):
        self.seed(self.today-timedelta(days=1),900,'Yesterday')
        self.seed(self.today,3600,'Today task');self.open()
        self.assertEqual(self.window.history_calendar.selected,self.today)
        self.assertEqual(self.window.history.count(),1)
        row=self.window.history.item(0).text()
        self.assertIn('｜60分钟｜Today task｜已结束',row)
        self.wait_for(lambda:'Today task' in self.window.details.toPlainText())
    def test_pick_day_survives_refresh_and_stale_details_are_ignored(self):
        yesterday=self.today-timedelta(days=1)
        old=self.seed(yesterday,900,'Yesterday');current=self.seed(self.today,3600,'Today')
        self.open();self.window.history_calendar.cells[yesterday].click()
        self.wait_for(lambda:'Yesterday' in self.window.details.toPlainText())
        self.window.show_history_detail(current,'STALE')
        self.assertNotIn('STALE',self.window.details.toPlainText())
        self.window.worker.submit('images',True)
        self.wait_for(lambda:self.window.images.isChecked())
        self.assertEqual(self.window.history_calendar.selected,yesterday)
        self.assertEqual(self.window.history.item(0).data(256),old)
        self.window.history_calendar.cells[self.today].click()
        self.wait_for(lambda:'Today' in self.window.details.toPlainText())
        self.assertEqual(self.window.history_calendar.selected,self.today)
    def test_empty_day_clears_list_and_previous_details(self):
        self.seed(self.today,600,'Today');self.open()
        self.wait_for(lambda:'Today' in self.window.details.toPlainText())
        self.window.history_calendar.cells[self.today-timedelta(days=1)].click()
        self.assertEqual(self.window.history.count(),0)
        self.assertEqual(self.window.details.toPlainText(),'')
        self.assertFalse(self.window.delete_history.isEnabled())
        self.assertFalse(self.window.history_empty.isHidden())
        self.assertIn('0分钟',self.window.history_day_summary.text())
    def test_delete_selected_session_clears_heat_and_keeps_day_selection(self):
        yesterday=self.today-timedelta(days=1);self.seed(yesterday,1800,'Delete me');self.open()
        self.window.history_calendar.cells[yesterday].click()
        self.wait_for(lambda:self.window.delete_history.isEnabled())
        self.window.delete_history.click()
        self.wait_for(lambda:self.window.history.count()==0)
        self.wait_for(lambda:yesterday.isoformat() not in self.window.history_payload['days'])
        self.assertEqual(self.window.history_calendar.selected,yesterday)

    def test_detail_refresh_preserves_scroll_for_same_task(self):
        sid=self.seed(self.today,600,'Today');self.open()
        self.wait_for(lambda:'Today' in self.window.details.toPlainText())
        text='\n'.join(f'Log line {i}' for i in range(200))
        self.window.show_history_detail(sid,text);self.app.processEvents()
        scroll=self.window.details.verticalScrollBar()
        scroll.setValue(scroll.maximum()//2);position=scroll.value()
        self.window.show_history_detail(sid,text+'\nMore lines')
        self.assertEqual(scroll.value(),position)
    def test_cell_tooltip_colors_future_days_and_month_rollover(self):
        widget=HistoryCalendar();start,end=month_window(date(2026,11,15))
        data={'today':'2026-11-15','start':start.isoformat(),'end':end.isoformat(),
              'days':{'2026-11-14':{'seconds':3600,'tasks':2}}}
        widget.set_data(data)
        self.assertIn('60分钟',widget.cells[date(2026,11,14)].toolTip())
        self.assertIn(COLORS[3],widget.cells[date(2026,11,14)].styleSheet())
        self.assertNotIn(date(2026,11,16),widget.cells)
        self.assertEqual(max(widget.cells),date(2026,11,15))
        from PySide6.QtWidgets import QPushButton
        self.assertEqual(widget.findChildren(QPushButton),[])
        widget.select(date(2026,6,10))
        widget.set_data({'today':'2026-12-01','start':'2026-07-01','end':'2027-01-01','days':{}})
        self.assertEqual(widget.selected,date(2026,12,1))
        widget.select(date(2026,10,15))
        widget.set_data({'today':'2026-12-02','start':'2026-07-01','end':'2027-01-01','days':{}})
        self.assertEqual(widget.selected,date(2026,10,15))
        widget.deleteLater();self.app.processEvents()
