package net.svrnet.bridge;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Set;

/**
 * IBKR as tools/ibkr/test/levels.py's stand-in (Broker there): the same shares, closes and refusals, orders kept
 * between connections, positions from the fills. The hold-book checks (HoldsCheck) run the Python suite's
 * scenarios against it.
 */
final class FakeBroker implements Broker {
    static final Map<String, Integer> CONIDS = Map.ofEntries(Map.entry("AAA", 1), Map.entry("BBB", 2), Map.entry("CCC", 3), Map.entry("DDD", 4),
            Map.entry("EEE", 5), Map.entry("XXX", 6), Map.entry("MC", 7), Map.entry("BRK B", 8), Map.entry("BHP", 9), Map.entry("D05", 10), Map.entry("SPY", 11));
    static final Map<String, Double> CLOSES = Map.ofEntries(Map.entry("AAA", 350.0), Map.entry("BBB", 120.0), Map.entry("CCC", 80.0), Map.entry("DDD", 100.0),
            Map.entry("EEE", 50.0), Map.entry("XXX", 100.0), Map.entry("MC", 500.0), Map.entry("BRK B", 480.0), Map.entry("BHP", 45.0), Map.entry("D05", 50.0),
            Map.entry("SPY", 600.0));

    /** An order at the stand-in: what was placed and where it stands. */
    static final class T {
        final Share share;
        final NewOrder order;
        final int id;
        String status = "Submitted";
        double filled;
        final List<String> log = new ArrayList<>();

        T(Share share, NewOrder order, int id) {
            this.share = share;
            this.order = order;
            this.id = id;
        }
    }

    final List<T> trades = new ArrayList<>();
    final List<Fill> fills = new ArrayList<>();
    Set<String> noCloseOrders = Set.of();
    private int nextId = 100;

    private static String symOf(int conId) {
        for (Map.Entry<String, Integer> e : CONIDS.entrySet()) if (e.getValue() == conId) return e.getKey();
        return "";
    }

    @Override
    public Share qualify(Share s) {
        String sym = s.symbol() != null && !s.symbol().isEmpty() ? s.symbol() : symOf(s.conId());
        if (!CONIDS.containsKey(sym)) return null;
        String ex = s.exchange() == null || s.exchange().isEmpty() ? "LSE" : s.exchange();
        return new Share(CONIDS.get(sym), sym, ex, "LSE", "GBP");
    }

    @Override
    public Share exitShare(Share s) {
        Share q = qualify(new Share(s.conId(), s.conId() != 0 ? "" : s.symbol(), "", "", ""));
        return q == null ? null : q.routed(q.primaryExchange());
    }

    @Override
    public Ticks ticks(Share s) {
        return new Ticks(null, 0.5, 1.0);
    }

    @Override
    public double referenceClose(Share s) {
        return CLOSES.get(s.symbol());
    }

    @Override
    public double floorPrice(Share s) {
        double lo = CLOSES.get(s.symbol()) * 0.97;
        return Holds.roundTo(lo, 0.5, "down");
    }

    @Override
    public List<Position> positions() {
        Map<Integer, Object[]> held = new LinkedHashMap<>();
        for (Fill f : fills) {
            Object[] h = held.computeIfAbsent(f.share().conId(), k -> new Object[] {f.share(), 0.0});
            h[1] = (double) h[1] + (f.side().equals("BOT") ? f.shares() : -f.shares());
        }
        List<Position> r = new ArrayList<>();
        for (Object[] h : held.values()) if ((double) h[1] > 0) r.add(new Position((Share) h[0], (double) h[1], 0));
        return r;
    }

    @Override
    public List<Holding> portfolio() {
        List<Holding> r = new ArrayList<>();
        for (Position p : positions()) r.add(new Holding(p.share(), p.qty(), CLOSES.get(p.share().symbol()), 0, p.avgCost()));
        return r;
    }

    private Ord ord(T t) {
        NewOrder o = t.order;
        return new Ord(t.id, t.id, o.parentId(), t.share, o.action(), o.type(), o.qty(), o.limit(), o.aux(), o.trailPercent(), o.trailStop(),
                o.ref(), o.tif(), o.goodTill(), t.status, t.filled, o.qty() - t.filled);
    }

    List<T> live() {
        List<T> r = new ArrayList<>();
        for (T t : trades) if (!t.status.equals("Filled") && !t.status.equals("Cancelled") && !t.status.equals("Inactive")) r.add(t);
        return r;
    }

    @Override
    public List<Ord> openOrders() {
        List<Ord> r = new ArrayList<>();
        for (T t : live()) r.add(ord(t));
        return r;
    }

    @Override
    public List<Ord> completedOrders() {
        return List.of();
    }

    @Override
    public List<Fill> fills() {
        return new ArrayList<>(fills);
    }

    @Override
    public int place(Share s, NewOrder o) {
        T t = new T(s, o, ++nextId);
        trades.add(t);
        if ((o.type().equals("LOC") || o.type().equals("MOC")) && noCloseOrders.contains(s.symbol())) {
            t.status = "Cancelled";
            t.log.add("387: Unsupported order type for this exchange and security type.");
        }
        if (s.symbol().equals("CCC") && "GTD".equals(o.tif())) {
            t.status = "Cancelled";
            t.log.add("10052: Invalid time in force: GTD is not allowed for this order");
        }
        return t.id;
    }

    @Override
    public void cancel(int orderId) {
        for (T t : trades) if (t.id == orderId && !t.status.equals("Filled") && !t.status.equals("Cancelled")) t.status = "Cancelled";
    }

    @Override
    public Fate fate(int orderId, int seconds) {
        for (T t : trades) {
            if (t.id != orderId) continue;
            String f = t.status.equals("Filled") ? "filled" : t.status.equals("Cancelled") || t.status.equals("Inactive") ? "dead" : "live";
            return new Fate(f, new ArrayList<>(t.log));
        }
        return new Fate("dead", List.of());
    }

    @Override
    public void sleep(double seconds) {
    }

    // the market
    void fill(T t, double qty, double price) {
        t.filled += qty;
        if (t.filled >= t.order.qty()) t.status = "Filled";
        fills.add(new Fill(t.share, t.order.action().equals("BUY") ? "BOT" : "SLD", qty, price, t.order.ref(), t.id, "e" + fills.size(), "", 0, 0));
    }

    T find(String sym, String action) {
        for (T t : trades) if (t.share.symbol().equals(sym) && t.order.action().equals(action)) return t;
        return null;
    }

    List<T> where(java.util.function.Predicate<T> p) {
        List<T> r = new ArrayList<>();
        for (T t : trades) if (p.test(t)) r.add(t);
        return r;
    }

    static final Map<String, Object> NONE = new HashMap<>();
}
