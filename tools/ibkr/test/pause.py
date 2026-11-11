"""The daily-bars fetch gives way to the bridge's other steps (no IB Gateway, no network):

  python3 tools/ibkr/test/pause.py

1. submit.py --bars --pause, on the stand-in IBKR of levels.py: while the pause file exists it asks IBKR for
   nothing, says so with the .ack file, and goes on (every share fetched) once the file is gone.
2. bridge.py's run_submit: a step (here --report) pauses a running fetch and waits for its .ack before it
   runs, and lets it go on after; an all-day watch (--rebuy) never pauses it; two steps at once keep the
   pause until the last one ends; and no fetch starts while a step runs (steps_pending).
"""
import json
import os
import sys
import tempfile
import threading
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import levels  # noqa: E402  the stand-in ib_async and the clock (it installs them)

submit = levels.submit


def fetch_pauses():
    d = tempfile.mkdtemp()
    book = {'asOf': '2026-11-11', 'orders': [], 'pool': ['AAA', 'BBB', 'CCC']}
    path, out, pause = os.path.join(d, 'UK.json'), os.path.join(d, 'bars.csv'), os.path.join(d, 'bars.pause')
    json.dump(book, open(path, 'w'))
    open(pause, 'w').close()  # a step holds the gateway when the fetch starts
    b = levels._broker(levels.Broker)
    b.events = []  # this test's own day: none of the other scenarios' events
    asked = []
    real = b.reqHistoricalData

    def counted(c, *a, **k):
        asked.append((c.symbol, os.path.exists(pause)))
        close = real(c, *a, **k)[0].close
        return [levels.NS(date=levels._dt.date(2026, 11, 11), open=close, high=close, low=close, close=close, volume=1000)]
    b.reqHistoricalData = counted
    sleeps = []
    real_sleep = time.sleep
    submit.time = time  # daily_bars imports time itself: shorten its 10.5 s spacing for the test
    time.sleep = lambda s: sleeps.append(s)
    try:
        done = {}
        def go():
            try:
                done['out'] = levels.run(['--market', 'UK', '--orders', path, '--bars', '--out', out, '--pause', pause])
            except BaseException as e:  # noqa: BLE001
                import traceback
                done['out'] = traceback.format_exc()
        th = threading.Thread(target=go)
        th.start()
        for _ in range(500):
            if os.path.exists(pause + '.ack'):
                break
            real_sleep(0.01)
        assert os.path.exists(pause + '.ack'), 'the fetch did not say it paused'
        assert not asked, f'IBKR asked while paused: {asked}'
        os.remove(pause)  # the step is done
        th.join(30)
        assert not th.is_alive(), 'the fetch did not go on'
    finally:
        time.sleep = real_sleep
    assert [s for s, _ in asked] == ['AAA', 'BBB', 'CCC'] and not any(p for _, p in asked), (asked, done.get('out'))
    assert not os.path.exists(pause + '.ack'), 'the .ack left behind'
    return done['out']


def bridge_waits():
    os.environ['ORDERS_DIR'] = tempfile.mkdtemp()
    sys.path.insert(0, os.path.join(HERE, '..'))
    import bridge
    calls = []
    bridge.run_submit_now = lambda args, timeout=300: (calls.append((list(args), os.path.exists(bridge.BARS_PAUSE), os.path.exists(bridge.BARS_PAUSE + '.ack'))), (0, 'ok'))[1]
    bridge.BARS_RUNNING.add('EU')  # a fetch running: it answers a pause after a moment
    stop = threading.Event()

    def fetch():
        while not stop.is_set():
            if os.path.exists(bridge.BARS_PAUSE) and not os.path.exists(bridge.BARS_PAUSE + '.ack'):
                time.sleep(0.3)  # its request in flight finishes first
                open(bridge.BARS_PAUSE + '.ack', 'w').close()
            time.sleep(0.05)
    th = threading.Thread(target=fetch, daemon=True)
    th.start()
    bridge.run_submit(['--market', 'US', '--report'])
    assert calls[-1] == (['--market', 'US', '--report'], True, True), calls  # ran paused, the fetch confirmed
    assert not os.path.exists(bridge.BARS_PAUSE), 'the fetch left paused after the step'
    bridge.run_submit(['--market', 'UK', '--rebuy'])
    assert calls[-1][1] is False, 'an all-day watch paused the fetch'
    # two steps at once: the pause holds until the last one ends
    assert bridge.bars_quiet(wait=5)
    bridge.run_submit(['--market', 'EU', '--close'])
    assert os.path.exists(bridge.BARS_PAUSE), 'the pause ended while another step still held it'
    bridge.bars_resume()
    assert not os.path.exists(bridge.BARS_PAUSE)
    stop.set()
    # a fetch does not start while a step runs (here a market's report)
    bridge.RUNNING.add(('US', 'report'))
    assert bridge.steps_pending(), 'a fetch could start during a report'
    bridge.RUNNING.discard(('US', 'report'))
    bridge.RUNNING.add(('US', 'bars'))
    assert not bridge.steps_pending() or any(True for _ in bridge.AUTO), 'a fetch held up by another fetch'
    bridge.RUNNING.discard(('US', 'bars'))
    return 'bridge: a step pauses the fetch and waits for it; a watch does not; the last step lets it go on; no fetch starts during a step'


if __name__ == '__main__':
    print(fetch_pauses())
    print(bridge_waits())
    print('pause: the daily-bars fetch gives way to the bridge\'s other steps; OK')
