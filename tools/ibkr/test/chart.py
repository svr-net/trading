"""The bridge's chart endpoint's bars (no IB Gateway, no network):

  python3 tools/ibkr/test/chart.py

bridge.chart_bars reads the kept IBKR bars file (ibkr-<MARKET>.csv): a share's last n bars, oldest first, found
by its ticker as the orders give it (EXCHANGE:CODE) or its code alone; none for a share not in the file.
"""
import csv
import os
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
os.environ['ORDERS_DIR'] = os.environ['HISTORY_DIR'] = tempfile.mkdtemp()
sys.path.insert(0, os.path.join(HERE, '..'))
import bridge  # noqa: E402

bridge.store_get = lambda *_: None  # no store here
with open(os.path.join(bridge.HISTORY, 'ibkr-EU.csv'), 'w', newline='') as f:
    w = csv.writer(f)
    w.writerow(['date', 'ticker', 'open', 'high', 'low', 'close', 'volume'])
    for d in range(1, 31):
        w.writerow([f'2026-10-{d:02d}', 'AMS:ASM', 100 + d, 102 + d, 99 + d, 101 + d, 1000 * d])
        w.writerow([f'2026-10-{d:02d}', 'FRA:RHM', 50, 51, 49, 50, 10])
bars = bridge.chart_bars('EU', 'AMS:ASM', 5)
assert [b['d'] for b in bars] == ['2026-10-26', '2026-10-27', '2026-10-28', '2026-10-29', '2026-10-30'], bars
assert bars[-1] == {'d': '2026-10-30', 'o': 130.0, 'h': 132.0, 'l': 129.0, 'c': 131.0, 'v': 30000.0}, bars[-1]
assert len(bridge.chart_bars('EU', 'ASM', 120)) == 30, 'the code alone did not find it'
assert bridge.chart_bars('EU', 'AMS:NONE', 5) == [], 'a share not in the file'
assert bridge.chart_bars('UK', 'TRB', 5) == [], 'a market with no file and no store'
import json
calls = []
def fake_store(path):
    calls.append(path)
    return json.dumps([{'code': 'ASM', 'sector': 'Technology', 'industry': 'Semiconductors'}]).encode() if path == 'profiles/AMS.json' else None
bridge.store_get = fake_store
assert bridge.profile_of('EU', 'AMS:ASM') == {'exchange': 'AMS', 'sector': 'Technology', 'industry': 'Semiconductors'}, bridge.profile_of('EU', 'AMS:ASM')
assert bridge.profile_of('EU', 'AMS:ASM')['sector'] == 'Technology' and calls.count('profiles/AMS.json') == 1, 'the profiles not kept between charts'
assert bridge.profile_of('US', 'TEAM') == {}, 'a share with no profile'
print('chart: a share\'s sector and industry from the store\'s profiles, read once; none when it has none; OK')
print('chart: a share\'s last bars from the kept IBKR file, by its ticker or code; none when it is not there; OK')
