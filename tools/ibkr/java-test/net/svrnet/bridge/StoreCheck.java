package net.svrnet.bridge;

import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Map;

/** IBKR's daily bars in the store by month (Store.byMonth, Store.join): split and put back together, the same file byte for byte. */
final class StoreCheck {
    static void check(boolean ok, Object why) {
        if (!ok) throw new AssertionError(String.valueOf(why));
    }

    public static void main(String[] args) throws Exception {
        List<String[]> rows = new ArrayList<>();
        rows.add(new String[] {"date", "ticker", "open", "high", "low", "close", "volume"});
        String[] days = {"2026-09-29", "2026-09-30", "2026-10-01", "2026-10-30", "2026-11-02", "2026-11-27"};
        for (String d : days)
            for (String t : new String[] {"AAA", "B,B", "CCC"}) rows.add(new String[] {d, t, "1.5", "2.0", "1.0", "1.75", "1000.0"});
        Path f = Files.createTempFile("daily", ".csv");
        Csv.write(f, rows);
        byte[] whole = Files.readAllBytes(f);
        Map<String, byte[]> m = Store.byMonth(rows);
        check(m.keySet().equals(new java.util.TreeSet<>(List.of("2026-09", "2026-10", "2026-11"))), "months " + m.keySet());
        for (byte[] b : m.values()) check(new String(b).startsWith("date,ticker,"), "each month starts with the header");
        byte[] back = Store.join(new ArrayList<>(m.values()));
        check(Arrays.equals(whole, back), "joined:\n" + new String(back) + "\nwhole:\n" + new String(whole));
        check(Csv.read(f).size() == rows.size(), "rows read back");
        check(Store.byMonth(rows.subList(0, 1)).isEmpty() && Store.join(List.of()) == null, "empty input");
        Files.delete(f);
        System.out.println("store by month: the bars split by month and put back together are the same file byte for byte (a quoted ticker too); OK");
    }

    private StoreCheck() {
    }
}
