"""Bridge new Windows zone IDs not yet recognized by the locked tzlocal map."""
from __future__ import annotations

import sys
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from tzlocal import get_localzone_name as _tzlocal_name

# Unicode CLDR common/supplemental/windowsZones.xml (2026-10-22).
# Do not guess an unknown zone from today's UTC offset: DST rules can differ.
_NEW_WINDOWS_ZONES = {
    'Alberta Standard Time': 'America/Edmonton',
    'British Columbia Standard Time': 'America/Vancouver',
}


def _windows_settings():
    import winreg

    with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE,
                        r'SYSTEM\CurrentControlSet\Control\TimeZoneInformation') as key:
        name = winreg.QueryValueEx(key, 'TimeZoneKeyName')[0].split('\0', 1)[0]
        try:
            disabled = winreg.QueryValueEx(key, 'DynamicDaylightTimeDisabled')[0]
        except FileNotFoundError:
            disabled = 0
        bias = winreg.QueryValueEx(key, 'Bias')[0] if disabled else None
    return name, bool(disabled), bias


def get_localzone_name() -> str:
    try:
        return _tzlocal_name()
    except ZoneInfoNotFoundError:
        if sys.platform != 'win32':
            raise
        windows_name, dst_disabled, bias = _windows_settings()
        name = _NEW_WINDOWS_ZONES.get(windows_name)
        if name is None:
            raise  # Keep unknown configurations visible, never silently use UTC.
        if dst_disabled:
            if not isinstance(bias, int) or bias % 60 or abs(bias) > 14 * 60:
                raise ValueError('Windows fixed time-zone bias is unsupported')
            hours = bias // 60  # Windows Bias and Etc/GMT both use the inverse sign.
            name = 'Etc/GMT' if not hours else f'Etc/GMT{hours:+d}'
        ZoneInfo(name)  # Fail explicitly if the bundled IANA data is missing.
        return name
