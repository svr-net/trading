package net.svrnet.bridge;

import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.StandardCopyOption;
import java.time.LocalDate;
import java.time.ZonedDateTime;
import java.time.format.DateTimeFormatter;
import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.stream.Stream;

/**
 * The bridge's state kept in the private store too (svr-net/eod: bridge/state.json), so the NAS folder is disposable:
 * the order history (history.jsonl), each multi-day book's positions and stops (holds-<MARKET>.json), the buying
 * switch's last value (buys-on), the model checks recorded (model-*) and the step marks of the last 7 days (state/).
 * Uploaded within a minute of any change (one file, one commit); at start, an empty history folder is restored from it.
 * A reset (POST /reset) archives the folder and writes an empty state, so a reset is never undone by a restore.
 * IBKR's daily bars, the auctions and the paths are stored on their own (ibkr/, auction/, paths/) and not mirrored.
 */
final class Mirror {
    static final String PATH = "bridge/state.json";
    static Path DIR = Path.of(Book.HISTORY);  // the checks point it elsewhere
    static final boolean ON = !Bridge.env("STATE_MIRROR", "1").equals("0");
    static volatile boolean dirty;
    static volatile String note = "not uploaded yet";
    static volatile Boolean ok;
    static final Object LOCK = new Object();

    static void changed() {
        dirty = true;
    }

    static boolean kept(Path rel) {
        final String f = rel.toString().replace('\\', '/');
        if (f.equals("history.jsonl") || f.equals("buys-on")) return true;
        if (!f.contains("/") && (f.startsWith("holds-") && f.endsWith(".json") || f.startsWith("model-"))) return true;
        if (f.startsWith("state/")) {
            final String[] p = f.split("/");
            if (p.length != 3) return false;
            try {
                return !LocalDate.parse(p[1]).isBefore(LocalDate.now().minusDays(7));
            } catch (RuntimeException e) {
                return false;
            }
        }
        return false;
    }

    /** The state as one JSON document: {version, at, files: {relative path: content}}. */
    static String snapshot() throws IOException {
        final Map<String, Object> files = new LinkedHashMap<>();
        if (Files.isDirectory(DIR)) {
            final List<Path> all = new ArrayList<>();
            try (Stream<Path> w = Files.walk(DIR, 3)) {
                w.filter(Files::isRegularFile).forEach(all::add);
            }
            all.sort(null);
            for (Path p : all) {
                final Path rel = DIR.relativize(p);
                if (kept(rel)) files.put(rel.toString().replace('\\', '/'), Files.readString(p, StandardCharsets.UTF_8));
            }
        }
        final Map<String, Object> doc = new LinkedHashMap<>();
        doc.put("version", 1);
        doc.put("at", ZonedDateTime.now().format(DateTimeFormatter.ISO_OFFSET_DATE_TIME));
        doc.put("bridge", Bridge.VERSION);
        doc.put("files", files);
        return Json.write(doc);
    }

    /** Uploads the state when it changed since the last upload; a failure keeps it to upload again. */
    static void upload() {
        if (!ON || Store.TOKEN.isEmpty() || !dirty) return;
        synchronized (LOCK) {
            dirty = false;
            try {
                final String s = snapshot();
                Store.put(PATH, s.getBytes(StandardCharsets.UTF_8), "bridge state " + ZonedDateTime.now().format(DateTimeFormatter.ofPattern("yyyy-MM-dd HH:mm z")));
                ok = true;
                note = "uploaded " + ZonedDateTime.now().format(DateTimeFormatter.ofPattern("yyyy-MM-dd HH:mm z")) + " (" + s.length() / 1024 + " kB)";
            } catch (IOException | RuntimeException e) {
                dirty = true;
                ok = false;
                note = "upload failed (" + e.getMessage() + "): tried again in a minute";
                System.err.println("bridge: state mirror: " + note);
            }
        }
    }

    static boolean emptyHere() {
        if (Files.exists(DIR.resolve("history.jsonl"))) return false;
        if (Files.isDirectory(DIR.resolve("state"))) return false;
        try (Stream<Path> l = Files.list(DIR)) {
            return l.noneMatch(p -> p.getFileName().toString().startsWith("holds-"));
        } catch (IOException e) {
            return true;  // no folder yet
        }
    }

    /** At start: an empty history folder is restored from the store's copy (never over files already here). */
    @SuppressWarnings("unchecked")
    static String restore() {
        if (!ON || Store.TOKEN.isEmpty()) return "off";
        if (!emptyHere()) return "the history folder has the state: kept";
        try {
            final byte[] body = Store.get(PATH);
            if (body == null) return "no copy in the store yet";
            return apply(new String(body, StandardCharsets.UTF_8));
        } catch (IOException | RuntimeException e) {
            return "not restored: " + e.getMessage();
        }
    }

    /** Writes a snapshot's files here (only the kept kinds, never outside the folder). */
    @SuppressWarnings("unchecked")
    static String apply(String snapshot) throws IOException {
        final Map<String, Object> doc = JsonParse.object(snapshot);
        final Object f = doc.get("files");
        if (!(f instanceof Map) || ((Map<String, Object>) f).isEmpty()) return "the store's copy is empty (a reset): nothing restored";
        int n = 0;
        for (Map.Entry<String, Object> e : ((Map<String, Object>) f).entrySet()) {
            final Path rel = Path.of(e.getKey()).normalize();
            if (rel.isAbsolute() || rel.startsWith("..") || !kept(rel)) continue;
            final Path out = DIR.resolve(rel);
            Files.createDirectories(out.getParent());
            Files.writeString(out, String.valueOf(e.getValue()), StandardCharsets.UTF_8);
            n++;
        }
        return "restored " + n + " files from the store's copy of " + doc.get("at");
    }

    /** Archives the state here (into history/archive-<time>/; IBKR's bars stay) and empties the store's copy. */
    static String reset() throws IOException {
        synchronized (LOCK) {
            final Path archive = DIR.resolve("archive-" + ZonedDateTime.now().format(DateTimeFormatter.ofPattern("yyyyMMdd-HHmmss")));
            Files.createDirectories(archive);
            int n = 0;
            try (Stream<Path> l = Files.list(DIR)) {
                for (Path p : (Iterable<Path>) l::iterator) {
                    final String name = p.getFileName().toString();
                    if (name.startsWith("archive-") || name.startsWith("ibkr-") || p.equals(archive)) continue;
                    Files.move(p, archive.resolve(name), StandardCopyOption.ATOMIC_MOVE);
                    n++;
                }
            }
            dirty = true;
            upload();
            final String store = !ON || Store.TOKEN.isEmpty() ? "no copy in the store (mirror off or no EOD_TOKEN)"
                    : Boolean.TRUE.equals(ok) && !dirty ? "the store's copy is now empty (" + note + ")" : "the store's copy NOT emptied yet (" + note + ")";
            return "archived " + n + " entries in " + DIR.relativize(archive) + "; " + store;
        }
    }

    static void start() {
        final Thread t = new Thread(() -> {
            while (true) {
                Recordings.sleep(60);
                try {
                    upload();
                } catch (RuntimeException e) {
                    System.err.println("bridge: state mirror: " + e);
                }
            }
        }, "mirror");
        t.setDaemon(true);
        t.start();
        Runtime.getRuntime().addShutdownHook(new Thread(Mirror::upload, "mirror-last"));  // docker stop: the last changes too
    }

    private Mirror() {
    }
}
