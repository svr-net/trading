package net.svrnet.bridge;

import java.time.LocalDate;
import java.time.ZoneOffset;
import java.time.ZonedDateTime;
import java.time.format.DateTimeFormatter;
import java.util.ArrayList;
import java.util.Comparator;
import java.util.HashMap;
import java.util.LinkedHashMap;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.TreeSet;

/**
 * The day's other order steps (submit.py, ported): the leftovers (the model's shares held from earlier days:
 * listed, or sold at the next price), the close (today's orders cancelled, what they bought sold at the close),
 * the report (today's fills by share) and the state (the page's "Today at IBKR"). Each returns submit.py's
 * output: its lines, its last line (the summary) and its exit code.
 */
final class Steps {
    /** A step's result as submit.py printed it. */
    record Result(int code, List<String> lines, String last) {
        String out() {
            StringBuilder b = new StringBuilder();
            for (String l : lines) b.append("  ").append(l).append('\n');
            return b.append(last).toString();
        }
    }

    static final Map<String, Set<String>> CURRENCIES = Map.of("UK", Set.of("GBP"), "US", Set.of("USD"), "EU", Set.of("EUR", "NOK"),
            "APAC", Set.of("AUD", "SGD"), "INDEX", Set.of("EUR", "USD"));

    static Set<String> mine(String tickers, String market) {
        Set<String> m = new LinkedHashSet<>();
        for (String t : (tickers == null ? "" : tickers).split(",")) {
            if (t.isEmpty()) continue;
            m.add(Holds.ibSymbol(t.substring(t.lastIndexOf(':') + 1), market));
        }
        return m;
    }

    /** submit.py --leftovers [--check]: the model's shares still held from an earlier day (never the hold book's). */
    static Result leftovers(Broker ib, String market, String tickers, boolean check, LocalDate today) {
        Set<String> mine = mine(tickers, market);
        Set<String> cur = CURRENCIES.get(market);
        List<Broker.Position> held = new ArrayList<>();
        for (Broker.Position p : ib.positions()) if (p.qty() > 0 && mine.contains(p.share().symbol()) && (p.share().currency() == null || cur.contains(p.share().currency()))) held.add(p);
        Map<Integer, Double> bookBought = new HashMap<>();
        for (Broker.Fill f : ib.fills()) {
            if (f.ref() != null && f.ref().startsWith("ofm-" + market + "H-") && f.side().equals("BOT")) bookBought.merge(f.share().conId(), f.shares(), Double::sum);
        }
        held.removeIf(p -> p.qty() - bookBought.getOrDefault(p.share().conId(), 0.0) <= 0);
        List<String> lines = new ArrayList<>();
        if (check) {
            for (Broker.Position p : held) lines.add(p.share().symbol() + ": " + Holds.g(p.qty()) + " shares still held");
            return new Result(held.isEmpty() ? 0 : 1, lines, held.isEmpty() ? market + ": none of the model's shares held"
                    : market + ": " + held.size() + " of the model's shares still held");
        }
        Set<Integer> waiting = new LinkedHashSet<>();
        for (Broker.Ord t : ib.openOrders()) if (t.action().equals("SELL") && !t.dead()) waiting.add(t.share().conId());
        String day = "ofm-" + market + "-" + today;
        Map<Integer, Double> bought = new HashMap<>();
        for (Broker.Fill f : ib.fills()) if (f.ref() != null && f.ref().startsWith(day) && f.side().equals("BOT")) bought.merge(f.share().conId(), f.shares(), Double::sum);
        boolean okAll = true;
        int n = 0;
        for (Broker.Position p : held) {
            double qty = p.qty() - bought.getOrDefault(p.share().conId(), 0.0) - bookBought.getOrDefault(p.share().conId(), 0.0);
            if (qty <= 0) continue;
            n++;
            if (waiting.contains(p.share().conId())) {
                lines.add(p.share().symbol() + ": a sale is already waiting");
                continue;
            }
            String[] r = Holds.sellOut(ib, p.share(), qty, "ofm-" + market + "-leftover-" + today, false);
            okAll &= r[0].equals("true");
            lines.add(r[1]);
        }
        ib.sleep(2);
        return new Result(okAll ? 0 : 1, lines, market + " leftovers: " + n + " held from earlier days" + (lines.isEmpty() ? "" : ": " + String.join("; ", lines)));
    }

    /** submit.py --close: today's orders cancelled (confirmed), what they bought sold at the close (never twice). */
    static Result close(Broker ib, String market, LocalDate today) {
        String ref = "ofm-" + market + "-" + today;
        List<Broker.Ord> mine = new ArrayList<>();
        for (Broker.Ord t : ib.openOrders()) if (ref.equals(t.ref())) mine.add(t);
        for (Broker.Ord t : mine) ib.cancel(t.orderId());
        for (int i = 0; i < 20; i++) {
            boolean done = true;
            for (Broker.Ord t : ib.openOrders()) if (ref.equals(t.ref()) && !t.dead() && !t.status().equals("Filled")) done = false;
            if (done) break;
            ib.sleep(0.5);
        }
        Map<Integer, String> stuck = new LinkedHashMap<>();
        Set<Integer> selling = new LinkedHashSet<>();
        for (Broker.Ord t : ib.openOrders()) {
            if (t.dead() || t.status().equals("Filled") || t.status().equals("PendingCancel")) continue;
            if (ref.equals(t.ref())) stuck.put(t.share().conId(), t.share().symbol());
            if ((ref + "-close").equals(t.ref()) && t.action().equals("SELL")) selling.add(t.share().conId());
        }
        Map<Integer, Object[]> filled = new LinkedHashMap<>();
        for (Broker.Fill f : ib.fills()) {
            if (ref.equals(f.ref()) && f.side().equals("BOT")) {
                Object[] v = filled.computeIfAbsent(f.share().conId(), k -> new Object[] {f.share(), 0.0});
                v[1] = (double) v[1] + f.shares();
            }
        }
        Map<Integer, Double> held = new HashMap<>();
        for (Broker.Position p : ib.positions()) held.put(p.share().conId(), p.qty());
        boolean okAll = true;
        List<String> lines = new ArrayList<>();
        for (String sym : stuck.values()) {
            okAll = false;
            lines.add(sym + ": an order of today could not be cancelled: NOT SOLD");
        }
        for (Map.Entry<Integer, Object[]> e : filled.entrySet()) {
            Broker.Share share = (Broker.Share) e.getValue()[0];
            double qty = Math.min((double) e.getValue()[1], held.getOrDefault(e.getKey(), 0.0));
            if (qty > 0 && selling.contains(e.getKey())) {
                lines.add(share.symbol() + ": its sale at the close is already placed");
            } else if (qty > 0 && !stuck.containsKey(e.getKey())) {
                String[] r = Holds.sellOut(ib, share, qty, ref + "-close", true);
                okAll &= r[0].equals("true");
                lines.add(r[1]);
            }
        }
        ib.sleep(2);
        return new Result(okAll ? 0 : 1, lines, market + " close: " + (lines.isEmpty() ? "nothing held" : String.join("; ", lines)));
    }

    static String soldBy(String ref, String type) {
        if (ref != null && ref.endsWith("-close")) return "close";
        return type == null ? "bracket" : switch (type) { case "STP", "TRAIL" -> "stop"; case "LMT" -> "take-profit"; default -> "bracket"; };
    }

    /** submit.py --report: today's fills of the model's orders (the day's and the hold book's), a share at a time. */
    static Map<String, Object> report(Broker ib, String market, LocalDate today) {
        String ref = "ofm-" + market + "-" + today;
        Map<Long, String> kinds = new HashMap<>();
        List<Broker.Ord> all = new ArrayList<>(ib.openOrders());
        all.addAll(ib.completedOrders());
        for (Broker.Ord o : all) kinds.put(o.permId(), o.type());
        Map<String, Map<String, Object>> per = new LinkedHashMap<>();
        for (Broker.Fill f : ib.fills()) {
            if (f.ref() == null || !f.ref().startsWith(ref)) continue;
            Map<String, Object> e = per.computeIfAbsent(f.share().symbol(), k -> {
                Map<String, Object> m = new LinkedHashMap<>();
                m.put("ticker", f.share().symbol());
                m.put("currency", f.share().currency());
                m.put("bought", 0.0);
                m.put("buy_value", 0.0);
                m.put("sold", 0.0);
                m.put("sell_value", 0.0);
                m.put("pnl", 0.0);
                m.put("commission", 0.0);
                return m;
            });
            boolean buy = f.side().equals("BOT");
            e.put(buy ? "bought" : "sold", (double) e.get(buy ? "bought" : "sold") + f.shares());
            if (!buy) {
                e.put("soldBy", f.ref().endsWith("-close") ? "close" : soldBy(null, kinds.getOrDefault(f.permId(), "")));
                e.put("soldAt", f.time() == null || f.time().isEmpty() ? null : f.time());
            }
            e.put(buy ? "buy_value" : "sell_value", (double) e.get(buy ? "buy_value" : "sell_value") + f.shares() * f.price());
            if (f.commission() != 0 && Math.abs(f.commission()) < 1e300) e.put("commission", (double) e.get("commission") + f.commission());
            if (f.realizedPnl() != 0 && Math.abs(f.realizedPnl()) < 1e300) e.put("pnl", (double) e.get("pnl") + f.realizedPnl());
        }
        List<Object> out = new ArrayList<>();
        for (Map<String, Object> e : per.values()) {
            double bought = (double) e.get("bought"), sold = (double) e.get("sold");
            Double buy = bought != 0 ? (double) e.get("buy_value") / bought : null;
            Double sell = sold != 0 ? (double) e.get("sell_value") / sold : null;
            Map<String, Object> m = new LinkedHashMap<>();
            m.put("ticker", e.get("ticker"));
            m.put("currency", e.get("currency"));
            m.put("bought", bought);
            m.put("buy", buy);
            m.put("sold", sold);
            m.put("sell", sell);
            m.put("ret", buy != null && sell != null ? sell / buy - 1 : null);
            m.put("pnl", e.get("pnl"));
            m.put("commission", e.get("commission"));
            m.put("soldBy", e.get("soldBy"));
            m.put("soldAt", e.get("soldAt"));
            out.add(m);
        }
        Map<String, Object> r = new LinkedHashMap<>();
        r.put("date", today.toString());
        r.put("market", market);
        r.put("trades", out);
        return r;
    }

    /** submit.py --state: the live state of the day's trades and the model's holdings (the page's "Today at IBKR"). */
    static Map<String, Object> state(Broker ib, String market, String tickers, double capital, LocalDate today) {
        String ref = "ofm-" + market + "-" + today, href = "ofm-" + market + "H-" + today;
        Set<String> mine = mine(tickers, market);
        Map<String, Map<String, Object>> rows = new LinkedHashMap<>();
        java.util.function.Function<Broker.Share, Map<String, Object>> row = c -> rows.computeIfAbsent(c.symbol(), k -> {
            Map<String, Object> r = new LinkedHashMap<>();
            r.put("ticker", c.symbol());
            r.put("currency", c.currency());
            r.put("state", "");
            r.put("limit", null);
            r.put("stop", null);
            r.put("take", null);
            r.put("bought", 0.0);
            r.put("buy", null);
            r.put("sold", 0.0);
            r.put("sell", null);
            r.put("soldBy", "");
            r.put("held", 0.0);
            r.put("price", null);
            r.put("pnl", 0.0);
            r.put("open", null);
            r.put("fees", 0.0);
            r.put("feesKnown", false);
            r.put("ordered", null);
            r.put("note", "");
            return r;
        });
        Map<Long, Broker.Ord> orders = new HashMap<>();
        List<Broker.Ord> all = new ArrayList<>(ib.openOrders());
        all.addAll(ib.completedOrders());
        for (Broker.Ord o : all) {
            String r0 = o.ref() == null ? "" : o.ref();
            if (!r0.startsWith(ref) && !r0.startsWith(href)) continue;
            orders.put(o.permId(), o);
            Map<String, Object> r = row.apply(o.share());
            if (o.action().equals("BUY") && o.parentId() == 0 && o.type().equals("LMT")) {
                r.put("limit", fin(o.limit()));
                r.put("buyStatus", o.status());
                r.put("ordered", (r.get("ordered") == null ? 0.0 : (double) r.get("ordered")) + o.qty());
            } else if (o.action().equals("SELL") && (o.type().equals("STP") || o.type().equals("TRAIL"))) {
                r.put("stop", o.type().equals("STP") ? fin(o.aux()) : fin(o.trailStop()));
            } else if (o.action().equals("SELL") && o.type().equals("LMT") && o.parentId() != 0) {
                if (r.get("take") == null) r.put("take", fin(o.limit()));
            }
        }
        Map<String, double[]> bv = new HashMap<>();
        for (Broker.Fill f : ib.fills()) {
            String r0 = f.ref() == null ? "" : f.ref();
            if (!r0.startsWith(ref) && !r0.startsWith(href)) continue;
            Map<String, Object> r = row.apply(f.share());
            String k = f.side().equals("BOT") ? "bought" : "sold";
            r.put(k, (double) r.get(k) + f.shares());
            bv.computeIfAbsent(r.get("ticker") + "\u0000" + k, x -> new double[1])[0] += f.shares() * f.price();
            if (k.equals("sold")) {
                Broker.Ord o = orders.get(f.permId());
                String by = r0.endsWith("-close") ? "close" : o == null ? "sale" : switch (o.type()) { case "STP", "TRAIL" -> "stop"; case "LMT" -> "take-profit"; default -> "sale"; };
                TreeSet<String> s = new TreeSet<>();
                for (String x : String.valueOf(r.get("soldBy")).split(", ")) if (!x.isEmpty()) s.add(x);
                s.add(by);
                r.put("soldBy", String.join(", ", s));
            }
            if (f.commission() != 0 && Math.abs(f.commission()) < 1e300) {
                r.put("fees", (double) r.get("fees") + f.commission());
                r.put("feesKnown", true);
            }
        }
        for (Map.Entry<String, double[]> e : bv.entrySet()) {
            String[] tk = e.getKey().split("\u0000");
            Map<String, Object> r = rows.get(tk[0]);
            double q = (double) r.get(tk[1]);
            r.put(tk[1].equals("bought") ? "buy" : "sell", q != 0 ? e.getValue()[0] / q : null);
        }
        for (Map<String, Object> r : rows.values()) {
            if ((double) r.get("sold") != 0 && r.get("buy") != null && r.get("sell") != null) {
                double mag = 1.0;
                for (Broker.Fill f : ib.fills()) {
                    if (!f.share().symbol().equals(r.get("ticker"))) continue;
                    Broker.Ticks t = ib.ticks(f.share());
                    mag = t == null || t.magnifier() == 0 ? 1.0 : t.magnifier();
                    if (mag == 1.0 && r.get("ordered") != null && r.get("limit") != null && capital > 0) {
                        double k = (double) r.get("ordered") * (double) r.get("limit") / capital;
                        mag = k > 0 && Math.abs(Math.log(k / 100.0)) < Math.abs(Math.log(k)) ? 100.0 : 1.0;
                    }
                    break;
                }
                r.put("pnl", Math.min((double) r.get("sold"), (double) r.get("bought")) * ((double) r.get("sell") - (double) r.get("buy")) / mag - (double) r.get("fees"));
            }
        }
        for (Broker.Holding it : ib.portfolio()) {
            if (it.qty() <= 0 || (!mine.contains(it.share().symbol()) && !rows.containsKey(it.share().symbol()))) continue;
            Map<String, Object> r = row.apply(it.share());
            r.put("held", it.qty());
            r.put("price", fin(it.marketPrice()));
            r.put("open", fin(it.unrealizedPnl()));
            if ((double) r.get("bought") == 0) r.put("buy", fin(it.avgCost()));
        }
        Set<String> waiting = new LinkedHashSet<>();
        for (Broker.Ord t : ib.openOrders()) if (t.action().equals("SELL") && !t.dead()) waiting.add(t.share().symbol());
        for (Map<String, Object> r : rows.values()) {
            String st = String.valueOf(r.getOrDefault("buyStatus", ""));
            r.remove("buyStatus");
            double held = (double) r.get("held"), bought = (double) r.get("bought"), sold = (double) r.get("sold");
            boolean w = waiting.contains(String.valueOf(r.get("ticker")));
            if (held != 0 && bought == 0) r.put("state", "held from an earlier day" + (w ? ", sale waiting" : ""));
            else if (held != 0) r.put("state", "bought, held" + (w ? ", sale placed" : ""));
            else if (bought != 0 && sold >= bought) r.put("state", "sold (" + (String.valueOf(r.get("soldBy")).isEmpty() ? "sale" : r.get("soldBy")) + ")");
            else if (st.equals("Cancelled") || st.equals("ApiCancelled") || st.equals("Inactive")) r.put("state", st.equals("Cancelled") ? "not bought (expired)" : "refused by IBKR");
            else if (st.equals("Filled")) r.put("state", "bought");
            else if (!st.isEmpty() && !st.equals("null")) r.put("state", "waiting to buy");
            else r.put("state", "no order");
        }
        List<Map<String, Object>> shares = new ArrayList<>(rows.values());
        shares.sort(Comparator.<Map<String, Object>, Boolean>comparing(r -> (double) r.get("held") == 0)
                .thenComparing(r -> (double) r.get("bought") == 0).thenComparing(r -> String.valueOf(r.get("ticker"))));
        Map<String, Object> out = new LinkedHashMap<>();
        out.put("market", market);
        out.put("date", today.toString());
        out.put("at", ZonedDateTime.now(Holds.clock.withZone(java.time.ZoneId.systemDefault())).format(DateTimeFormatter.ISO_OFFSET_DATE_TIME).replaceAll("\\.\\d+", ""));
        out.put("shares", new ArrayList<Object>(shares));
        return out;
    }

    static Double fin(double x) {
        return Double.isFinite(x) && Math.abs(x) < 1e300 ? x : null;
    }

    private Steps() {
    }

    @SuppressWarnings("unused")
    private static final ZoneOffset UTC = ZoneOffset.UTC;
}
