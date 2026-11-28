package net.svrnet.bridge;

import java.util.concurrent.atomic.AtomicInteger;
import java.util.concurrent.atomic.AtomicIntegerArray;

/**
 * The bars fetches' workers (Recordings.parallel): every item done once, at most BARS_PARALLEL at a time, a pause
 * acknowledged only once no worker is busy (and nothing started while it holds), a worker's failure thrown.
 */
final class ParallelCheck {
    static void check(boolean ok, Object why) {
        if (!ok) throw new AssertionError(String.valueOf(why));
    }

    public static void main(String[] args) {
        // every item once, never more than BARS_PARALLEL at once
        int n = 40;
        AtomicIntegerArray seen = new AtomicIntegerArray(n);
        AtomicInteger now = new AtomicInteger(), most = new AtomicInteger();
        Recordings.Pause free = new Recordings.Pause();
        Recordings.parallel(n, free, i -> {
            int c = now.incrementAndGet();
            most.accumulateAndGet(c, Math::max);
            Recordings.sleep(0.02);
            seen.incrementAndGet(i);
            now.decrementAndGet();
        });
        for (int i = 0; i < n; i++) check(seen.get(i) == 1, "item " + i + " done " + seen.get(i) + " times");
        check(most.get() <= Recordings.BARS_PARALLEL && most.get() > 1, "at most " + Recordings.BARS_PARALLEL + " at once: " + most.get());

        // a pause: acknowledged only when no worker is busy; nothing starts while it holds
        Recordings.Pause pause = new Recordings.Pause();
        AtomicInteger busy = new AtomicInteger(), startedWhileHeld = new AtomicInteger(), done = new AtomicInteger();
        AtomicInteger ackedBusy = new AtomicInteger(-1);
        Thread step = new Thread(() -> {
            Recordings.sleep(0.15);
            pause.hold();
            for (int k = 0; k < 100 && !pause.acked(); k++) Recordings.sleep(0.05);
            ackedBusy.set(pause.acked() ? busy.get() : -1);
            int before = done.get();
            Recordings.sleep(0.5);
            if (done.get() != before) startedWhileHeld.incrementAndGet();
            pause.release();
        });
        step.start();
        Recordings.parallel(30, pause, i -> {
            busy.incrementAndGet();
            Recordings.sleep(0.05);
            busy.decrementAndGet();
            done.incrementAndGet();
        });
        try {
            step.join();
        } catch (InterruptedException e) {
            Thread.currentThread().interrupt();
        }
        check(ackedBusy.get() == 0, "the pause acknowledged with " + ackedBusy.get() + " busy");
        check(startedWhileHeld.get() == 0, "work done while the pause held");
        check(done.get() == 30, "all done after the pause: " + done.get());

        // a failure stops the rest and is thrown
        boolean thrown = false;
        try {
            Recordings.parallel(20, new Recordings.Pause(), i -> {
                if (i == 3) throw new IllegalStateException("boom");
                Recordings.sleep(0.01);
            });
        } catch (IllegalStateException e) {
            thrown = "boom".equals(e.getMessage());
        }
        check(thrown, "a worker's failure not thrown");
        System.out.println("bars workers: every item once, at most " + Recordings.BARS_PARALLEL + " at once, the pause acknowledged only when idle, a failure thrown; OK");
    }

    private ParallelCheck() {
    }
}
