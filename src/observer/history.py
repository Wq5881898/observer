"""Local-day history aggregation; no GUI or inference dependencies."""
from __future__ import annotations

from datetime import date, datetime, time, timedelta, timezone
import json
from zoneinfo import ZoneInfo


def month_window(today: date, months=6):
    if type(months) is not int or not 1 <= months <= 12:
        raise ValueError('历史月份范围无效')
    ordinal = today.year*12 + today.month-1
    year, month = divmod(ordinal-months+1, 12)
    start = date(year, month+1, 1)
    year, month = divmod(ordinal+1, 12)
    return start, date(year, month+1, 1)


def minutes_text(seconds):
    if seconds is None:
        return '时长未知'
    if 0 < seconds < 6:
        return '＜0.1分钟'
    return f'{seconds/60:.1f}'.rstrip('0').rstrip('.') + '分钟'


def heat_level(seconds):
    if seconds <= 0:
        return 0
    return 1 + sum(seconds >= boundary*60 for boundary in (30, 60, 120, 240))


def _mode(raw):
    try:
        data = json.loads(raw)
        if not isinstance(data, dict):
            return 'unknown'
        return 'demo' if data.get('source') == 'synthetic-demo' else 'real'
    except (ValueError, TypeError):
        return 'unknown'


def calendar_history(db, now: datetime, timezone_name: str):
    if now.tzinfo is None or now.utcoffset() is None:
        raise ValueError('历史日期必须包含时区')
    zone = ZoneInfo(timezone_name)
    today = now.astimezone(zone).date()
    start, end = month_window(today)
    begin_utc = datetime.combine(start, time.min, zone).astimezone(timezone.utc)
    end_utc = datetime.combine(end, time.min, zone).astimezone(timezone.utc)
    sessions = db.execute(
        'SELECT s.id,s.title,s.started,s.ended,s.updated,s.state,s.identity,t.eligible, '
        'EXISTS(SELECT 1 FROM session_work w WHERE w.session_id=s.id) AS tracked '
        'FROM sessions s LEFT JOIN totals t ON t.session_id=s.id AND t.signal=\'away\' '
        'WHERE (s.started>=? AND s.started<?) OR s.id IN('
        'SELECT session_id FROM session_work WHERE ended>=? AND started<?) '
        'ORDER BY s.started DESC',
        (begin_utc.isoformat(), end_utc.isoformat(),
         begin_utc.isoformat(), end_utc.isoformat())).fetchall()
    intervals = {}
    for row in db.execute('SELECT * FROM session_work WHERE started < ? AND ended >= ?',
                          (end_utc.isoformat(), begin_utc.isoformat())):
        intervals.setdefault(row['session_id'], []).append(row)
    days, records = {}, []
    for row in sessions:
        portions = {}
        day_starts = {}
        started = datetime.fromisoformat(row['started']).astimezone(zone)
        legacy_note = ''
        if row['tracked']:
            for interval in intervals.get(row['id'], []):
                a = datetime.fromisoformat(interval['started']).astimezone(timezone.utc)
                b = datetime.fromisoformat(interval['ended']).astimezone(timezone.utc)
                day = max(start, a.astimezone(zone).date())
                last = min(today, b.astimezone(zone).date())
                if a == b and start <= day < end and day <= today:
                    portions.setdefault(day, 0.)
                    day_starts.setdefault(day, a)
                while day <= last and day < end:
                    left = datetime.combine(day, time.min, zone).astimezone(timezone.utc)
                    right = datetime.combine(day+timedelta(days=1), time.min, zone).astimezone(timezone.utc)
                    overlap = (min(b, right)-max(a, left)).total_seconds()
                    if overlap > 0:
                        portions[day] = portions.get(day, 0.) + overlap
                        piece_start = max(a, left)
                        day_starts[day] = min(day_starts.get(day, piece_start), piece_start)
                    day += timedelta(days=1)
        elif start <= started.date() <= today:
            # Legacy totals already exclude paused/resting time. They cannot
            # locate pauses on either side of midnight, so never interpolate.
            portions[started.date()] = row['eligible']
            day_starts[started.date()] = started
            finished = datetime.fromisoformat(row['ended'] or row['updated']).astimezone(zone)
            if row['eligible'] is None:
                legacy_note = '旧记录未保存实际计时时长'
            elif finished.date() != started.date():
                legacy_note = '旧记录缺少跨日分段，时长计入开始日期'
        mode = _mode(row['identity'])
        for day, seconds in portions.items():
            key = day.isoformat()
            daily = days.setdefault(key, {'seconds': 0., 'tasks': 0, 'demo_tasks': 0,
                                         'unknown_tasks': 0, 'unknown_duration_tasks': 0,
                                         'legacy': False})
            if mode == 'real':
                daily['tasks'] += 1
                if seconds is not None:
                    daily['seconds'] += seconds
                else:
                    daily['unknown_duration_tasks'] += 1
            elif mode == 'demo':
                daily['demo_tasks'] += 1
            else:
                daily['unknown_tasks'] += 1
            daily['legacy'] |= bool(legacy_note)
            records.append({'id': row['id'], 'title': row['title'], 'state': row['state'],
                            'started': row['started'], 'day': key, 'seconds': seconds,
                            'day_started': day_starts[day].isoformat(),
                            'mode': mode, 'timing_note': legacy_note})
    return {'today': today.isoformat(), 'start': start.isoformat(), 'end': end.isoformat(),
            'days': days, 'rows': records}
