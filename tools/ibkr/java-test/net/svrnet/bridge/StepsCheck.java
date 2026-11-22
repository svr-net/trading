package net.svrnet.bridge;

import java.nio.file.Files;
import java.nio.file.Path;
import java.time.LocalDate;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;

/**
 * The day's other steps (Steps) on the stand-in IBKR (FakeBroker), and the chart's bars (Charts, Csv): what tools/ibkr/test/levels.py's
 * leftovers and close cases and tools/ibkr/test/chart.py checked in Python.
 */
final class StepsCheck {
    static final LocalDate TODAY = LocalDate.of(2026, 11, 23);

    static void check(boolean ok, Object why) {
        if (!ok) throw new AssertionError(String.valueOf(why));
    }

    static Broker.Share share(String sym) {
        return new FakeBroker().qualify(new Broker.Share(0, sym, "LSE", "", "GBP"));
    }

    static FakeBroker.T buy(FakeBroker b, String sym, double qty, String ref) {
        int id = b.place(share(sym), Broker.NewOrder.of("BUY", "LMT", qty, FakeBroker.CLOSES.get(sym), 0, ref, "DAY"));
        for (FakeBroker.T t : b.trades) if (t.id == id) return t;
        throw new AssertionError("not placed");
    }

    /** The model's shares from an earlier day sold at the next price; the hold book's never; a share whose sale waits not sold twice. */
    static String leftovers() {
        FakeBroker b = new FakeBroker();
        FakeBroker.T a = buy(b, "AAA", 10, "ofm-UK-2026-11-19");
        b.fill(a, 10, 350);
        FakeBroker.T h = buy(b, "BBB", 5, "ofm-UKH-2026-11-20");
        b.fill(h, 5, 120);
        Steps.Result c = Steps.leftovers(b, "UK", "AAA,BBB", true, TODAY);
        check(c.code() == 1 && c.last().equals("UK: 1 of the model's shares still held") && c.lines().equals(List.of("AAA: 10 shares still held")), c.out());
        Steps.Result s = Steps.leftovers(b, "UK", "AAA,BBB", false, TODAY);
        List<FakeBroker.T> sells = b.where(t -> t.order.action().equals("SELL"));
        check(s.code() == 0 && sells.size() == 1 && sells.get(0).share.symbol().equals("AAA") && sells.get(0).order.qty() == 10
                && sells.get(0).order.ref().equals("ofm-UK-leftover-" + TODAY) && s.last().startsWith("UK leftovers: 1 held from earlier days: AAA"), s.out());
        Steps.Result again = Steps.leftovers(b, "UK", "AAA,BBB", false, TODAY);
        check(b.where(t -> t.order.action().equals("SELL")).size() == 1 && again.out().contains("AAA: a sale is already waiting"), again.out());
        Steps.Result none = Steps.leftovers(new FakeBroker(), "UK", "AAA", true, TODAY);
        check(none.code() == 0 && none.last().equals("UK: none of the model's shares held"), none.out());
        return s.out() + "\n" + again.out() + "\n";
    }

    /** The close: today's orders cancelled, what they bought sold at the close (MOC), never twice. */
    static String close() {
        FakeBroker b = new FakeBroker();
        String ref = "ofm-UK-" + TODAY;
        FakeBroker.T c = buy(b, "CCC", 20, ref);
        b.fill(c, 12, 80);  // partly filled: the rest still working
        FakeBroker.T d = buy(b, "DDD", 7, ref);  // never filled
        Steps.Result r = Steps.close(b, "UK", TODAY);
        List<FakeBroker.T> sells = b.where(t -> t.order.action().equals("SELL"));
        check(c.status.equals("Cancelled") && d.status.equals("Cancelled"), c.status + " " + d.status);
        check(r.code() == 0 && sells.size() == 1 && sells.get(0).share.symbol().equals("CCC") && sells.get(0).order.qty() == 12
                && sells.get(0).order.type().equals("MOC") && sells.get(0).order.ref().equals(ref + "-close"), r.out());
        Steps.Result again = Steps.close(b, "UK", TODAY);
        check(b.where(t -> t.order.action().equals("SELL")).size() == 1 && again.out().contains("CCC: its sale at the close is already placed"), again.out());
        Steps.Result nothing = Steps.close(new FakeBroker(), "UK", TODAY);
        check(nothing.code() == 0 && nothing.last().equals("UK close: nothing held"), nothing.out());
        return r.out() + "\n" + again.out() + "\n";
    }

    /** The report: today's fills by share, the day's and the hold book's, with the return and how it was sold. */
    @SuppressWarnings("unchecked")
    static String report() {
        FakeBroker b = new FakeBroker();
        FakeBroker.T c = buy(b, "CCC", 10, "ofm-UK-" + TODAY);
        b.fill(c, 10, 80);
        int sid = b.place(share("CCC"), Broker.NewOrder.of("SELL", "MOC", 10, 0, 0, "ofm-UK-" + TODAY + "-close", "DAY"));
        for (FakeBroker.T t : b.trades) if (t.id == sid) b.fill(t, 10, 88);
        FakeBroker.T h = buy(b, "AAA", 3, "ofm-UKH-" + TODAY);
        b.fill(h, 3, 350);
        FakeBroker.T old = buy(b, "BBB", 4, "ofm-UK-2026-11-20");
        b.fill(old, 4, 120);
        Map<String, Object> r = Steps.report(b, "UK", TODAY);
        List<Map<String, Object>> t = (List<Map<String, Object>>) r.get("trades");
        check(t.size() == 1 && t.get(0).get("ticker").equals("CCC") && (double) t.get(0).get("bought") == 10 && (double) t.get(0).get("sold") == 10
                && Math.abs((double) t.get(0).get("ret") - 0.1) < 1e-12 && "close".equals(t.get(0).get("soldBy")), Json.write(r));
        return Json.write(r) + "\n";
    }

    /** IBKR's kept file read back (quoted tickers too), and the store's older bars joined on in IBKR's units (pence for pounds). */
    static String charts() throws Exception {
        Path dir = Files.createTempDirectory("charts-check");
        Path f = dir.resolve("x.csv");
        List<String[]> rows = new ArrayList<>();
        rows.add(new String[] {"date", "ticker", "open", "high", "low", "close", "volume"});
        rows.add(new String[] {"2026-10-01", "AMS:ASM", "1.0", "2.0", "0.5", "1.5", "10.0"});
        rows.add(new String[] {"2026-10-02", "A,B \"C\"", "1.0", "2.0", "0.5", "1.5", ""});
        Csv.write(f, rows);
        List<String[]> back = Csv.read(f);
        check(back.size() == 3 && back.get(2)[1].equals("A,B \"C\"") && back.get(2)[6].isEmpty() && back.get(1)[5].equals("1.5"), back.size());
        List<Map<String, Object>> ibkr = List.of(Charts.bar("2026-10-02", 101, 103, 100, 102, 5), Charts.bar("2026-10-03", 102, 104, 101, 103, 6));
        List<Map<String, Object>> store = List.of(Charts.bar("2026-10-01", 1.0, 1.0, 1.0, 1.0, 1), Charts.bar("2026-10-02", 1.02, 1.04, 1.0, 1.02, 1));
        List<Map<String, Object>> joined = Charts.withOlder(ibkr, store);
        check(joined.size() == 3 && joined.get(0).get("d").equals("2026-10-01") && Math.abs((double) joined.get(0).get("c") - 100.0) < 1e-9
                && (double) joined.get(1).get("c") == 102.0, joined);
        List<Map<String, Object>> other = List.of(Charts.bar("2026-10-02", 900000, 900000, 900000, 900000, 1), Charts.bar("2026-10-01", 900000, 900000, 900000, 900000, 1));
        check(Charts.withOlder(ibkr, other).size() == 2, "another share's prices joined on");
        check(Recordings.isoDay("20261120").equals("2026-11-20") && Recordings.py(12345678.0).equals("12345678.0") && Recordings.py(1.5).equals("1.5")
                && Recordings.py(null).isEmpty(), "numbers as Python writes them");
        check(Ib.hhmmUtc("20261120 09:31:05 Europe/London").equals("09:31") && Ib.hhmmUtc("20261120 09:31:05 US/Eastern").equals("14:31")
                && Ib.hhmmUtc("20261120-14:02:00").equals("14:02"), "fill times in UTC");
        return "";
    }

    public static void main(String[] args) throws Exception {
        System.out.print(leftovers());
        System.out.println("leftovers: the model's shares from earlier days sold at the next price, the hold book's never, a waiting sale not doubled; OK");
        System.out.print(close());
        System.out.println("close: today's orders cancelled, what they bought sold at the close, never twice; OK");
        System.out.print(report());
        System.out.println("report: today's fills by share with the return and how it was sold; OK");
        System.out.print(charts());
        System.out.println("charts: the kept CSV read back, the store's older bars in IBKR's units, numbers and times as before; OK");
    }

    private StepsCheck() {
    }
}
