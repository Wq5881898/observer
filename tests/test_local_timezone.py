from datetime import datetime, timedelta
import unittest
from unittest.mock import patch
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from observer.local_timezone import get_localzone_name


class LocalTimezoneTests(unittest.TestCase):
    def test_existing_zone_uses_tzlocal_without_registry_fallback(self):
        with patch('observer.local_timezone._tzlocal_name', return_value='Asia/Shanghai'), \
                patch('observer.local_timezone._windows_settings') as settings:
            self.assertEqual(get_localzone_name(), 'Asia/Shanghai')
            settings.assert_not_called()

    def test_new_windows_zones_follow_cldr_not_current_offset(self):
        for windows, iana in [('Alberta Standard Time', 'America/Edmonton'),
                              ('British Columbia Standard Time', 'America/Vancouver')]:
            with self.subTest(windows=windows), \
                    patch('observer.local_timezone.sys.platform', 'win32'), \
                    patch('observer.local_timezone._tzlocal_name', side_effect=ZoneInfoNotFoundError(windows)), \
                    patch('observer.local_timezone._windows_settings', return_value=(windows, False, None)):
                self.assertEqual(get_localzone_name(), iana)
        # Installed tzdata 2026.4 includes Alberta's post-transition UTC-6 rule.
        self.assertEqual(datetime(2026, 11, 2, 10, tzinfo=ZoneInfo('America/Edmonton')).utcoffset(),
                         timedelta(hours=-6))

    def test_disabled_dst_preserves_windows_standard_bias(self):
        for bias, name in [(420, 'Etc/GMT+7'), (-480, 'Etc/GMT-8'), (0, 'Etc/GMT')]:
            with self.subTest(bias=bias), \
                    patch('observer.local_timezone.sys.platform', 'win32'), \
                    patch('observer.local_timezone._tzlocal_name', side_effect=ZoneInfoNotFoundError('new')), \
                    patch('observer.local_timezone._windows_settings', return_value=('Alberta Standard Time', True, bias)):
                self.assertEqual(get_localzone_name(), name)
                self.assertEqual(datetime(2026, 11, 2, tzinfo=ZoneInfo(name)).utcoffset(),
                                 timedelta(minutes=-bias))

    def test_unknown_zone_is_not_guessed_or_replaced_with_utc(self):
        with patch('observer.local_timezone.sys.platform', 'win32'), \
                patch('observer.local_timezone._tzlocal_name', side_effect=ZoneInfoNotFoundError('unknown')), \
                patch('observer.local_timezone._windows_settings', return_value=('Unknown Future Time', False, None)):
            with self.assertRaises(ZoneInfoNotFoundError):
                get_localzone_name()

    def test_non_windows_errors_are_not_reinterpreted(self):
        with patch('observer.local_timezone.sys.platform', 'linux'), \
                patch('observer.local_timezone._tzlocal_name', side_effect=ZoneInfoNotFoundError('unknown')), \
                patch('observer.local_timezone._windows_settings') as settings:
            with self.assertRaises(ZoneInfoNotFoundError):
                get_localzone_name()
            settings.assert_not_called()

    def test_invalid_fixed_bias_is_rejected(self):
        with patch('observer.local_timezone.sys.platform', 'win32'), \
                patch('observer.local_timezone._tzlocal_name', side_effect=ZoneInfoNotFoundError('new')), \
                patch('observer.local_timezone._windows_settings', return_value=('Alberta Standard Time', True, 345)):
            with self.assertRaises(ValueError):
                get_localzone_name()
