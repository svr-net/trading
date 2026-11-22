package net.svrnet.bridge;

import java.io.IOException;
import java.io.InputStream;
import java.io.OutputStream;
import java.net.HttpURLConnection;
import java.net.URL;
import java.nio.charset.StandardCharsets;
import java.util.Base64;
import java.util.LinkedHashMap;
import java.util.Map;

/** The private store (svr-net/eod) through GitHub's contents API with EOD_TOKEN, as bridge.py's store_get and store_put. */
final class Store {
    static final String TOKEN = Bridge.env("EOD_TOKEN", Bridge.env("EOD_READ_TOKEN", ""));
    static final String BASE = "https://api.github.com/repos/svr-net/eod/contents/";

    static HttpURLConnection open(String url, String accept, int seconds) throws IOException {
        HttpURLConnection c = (HttpURLConnection) new URL(url).openConnection();
        c.setConnectTimeout(20_000);
        c.setReadTimeout(seconds * 1000);
        c.setRequestProperty("Authorization", "Bearer " + TOKEN);
        c.setRequestProperty("Accept", accept);
        c.setRequestProperty("User-Agent", "ofm-bridge");
        return c;
    }

    /** A store read's outcome: the bytes (null: not in the store) and whether they are the cached copy GitHub could not refresh. */
    record Read(byte[] body, boolean stale, String note) { }

    static final java.util.concurrent.ConcurrentHashMap<String, Object> LOCKS = new java.util.concurrent.ConcurrentHashMap<>();
    static volatile String lastStale = "";

    static java.nio.file.Path cacheFile(String path) {
        return java.nio.file.Path.of(Book.CACHE, "store", path.replace("..", "_"));
    }

    /** One file of the store, or null when it is not there (through the disk cache: see read). */
    static byte[] get(String path) throws IOException {
        return read(path).body();
    }

    /**
     * One file of the store through a disk cache (ORDERS_DIR/store/<path>, with GitHub's ETag beside it): asked with
     * If-None-Match, an unchanged file is a 304 (no download, no rate limit) and the cached copy is used; a changed one
     * replaces it. When GitHub does not answer (a timeout, a 5xx, the NAS offline), the last good copy is used and the
     * read says so (stale); only with no copy at all does it fail. A 404 removes the copy (the file is gone).
     */
    static Read read(String path) throws IOException {
        final java.nio.file.Path file = cacheFile(path), tag = java.nio.file.Path.of(file + ".etag");
        synchronized (LOCKS.computeIfAbsent(path, k -> new Object())) {
            final boolean have = java.nio.file.Files.exists(file);
            int code;
            HttpURLConnection c;
            try {
                c = open(BASE + path, "application/vnd.github.raw", 60);
                c.setRequestProperty("Cache-Control", "no-cache");
                if (have && java.nio.file.Files.exists(tag)) c.setRequestProperty("If-None-Match", java.nio.file.Files.readString(tag).strip());
                code = c.getResponseCode();
            } catch (IOException e) {
                if (have) return stale(path, file, "GitHub not reachable (" + e.getClass().getSimpleName() + ")");
                throw e;
            }
            if (code == 304 && have) return new Read(java.nio.file.Files.readAllBytes(file), false, "");
            if (code == 404) {
                java.nio.file.Files.deleteIfExists(file);
                java.nio.file.Files.deleteIfExists(tag);
                return new Read(null, false, "");
            }
            if (code == 401 || code == 403) throw new Http(code);  // the token: never hidden behind a cached copy
            if (code != 200) {
                if (have) return stale(path, file, "GitHub answered " + code);
                throw new Http(code);
            }
            byte[] body;
            try (InputStream in = c.getInputStream()) {
                body = in.readAllBytes();
            } catch (IOException e) {
                if (have) return stale(path, file, "the download broke off (" + e.getClass().getSimpleName() + ")");
                throw e;
            }
            java.nio.file.Files.createDirectories(file.getParent());
            final java.nio.file.Path tmp = java.nio.file.Path.of(file + ".tmp");
            java.nio.file.Files.write(tmp, body);
            java.nio.file.Files.move(tmp, file, java.nio.file.StandardCopyOption.REPLACE_EXISTING, java.nio.file.StandardCopyOption.ATOMIC_MOVE);
            final String etag = c.getHeaderField("ETag");
            if (etag != null) java.nio.file.Files.writeString(tag, etag);
            else java.nio.file.Files.deleteIfExists(tag);
            return new Read(body, false, "");
        }
    }

    private static Read stale(String path, java.nio.file.Path file, String why) throws IOException {
        lastStale = path + ": " + why + " at " + java.time.ZonedDateTime.now().format(java.time.format.DateTimeFormatter.ofPattern("yyyy-MM-dd HH:mm z"));
        System.err.println("bridge: store " + path + ": " + why + "; the cached copy is used");
        return new Read(java.nio.file.Files.readAllBytes(file), true, why);
    }

    /**
     * Writes one file to the store. A large file (IBKR's daily bars: 6-12 MB) takes minutes to upload from the NAS: up to 10 minutes, and a
     * timeout or a conflict (the file changed meanwhile) is tried again with the file's newest sha (18 Nov).
     */
    static void put(String path, byte[] data, String message) throws IOException {
        String content = Base64.getEncoder().encodeToString(data);
        IOException last = null;
        for (int attempt = 0; attempt < 3; attempt++) {
            try {
                String sha = null;
                HttpURLConnection g = open(BASE + path, "application/vnd.github+json", 60);
                int gc = g.getResponseCode();
                if (gc == 200) {
                    try (InputStream in = g.getInputStream()) {
                        Object s = JsonParse.object(new String(in.readAllBytes(), StandardCharsets.UTF_8)).get("sha");
                        sha = s == null ? null : s.toString();
                    }
                } else if (gc != 404) {
                    throw new Http(gc);
                }
                Map<String, Object> body = new LinkedHashMap<>();
                body.put("message", message);
                body.put("content", content);
                if (sha != null) body.put("sha", sha);
                HttpURLConnection p = open(BASE + path, "application/vnd.github+json", Math.max(60, 60 + content.length() / 25000));
                p.setRequestMethod("PUT");
                p.setDoOutput(true);
                p.setRequestProperty("Content-Type", "application/json");
                try (OutputStream o = p.getOutputStream()) {
                    o.write(Json.write(body).getBytes(StandardCharsets.UTF_8));
                }
                int pc = p.getResponseCode();
                if (pc == 200 || pc == 201) return;
                throw new Http(pc);
            } catch (Http e) {
                if (!(e.code == 409 || e.code == 422 || e.code >= 500) || attempt == 2) throw e;
                last = e;
            } catch (IOException e) {
                if (attempt == 2) throw e;
                last = e;
            }
            Recordings.sleep(10 * (attempt + 1));
        }
        throw last;
    }

    /** Whether the token reads svr-net/eod and may write it: true, false (no push), or an error's note. */
    static Object canPush() throws IOException {
        HttpURLConnection c = open("https://api.github.com/repos/svr-net/eod", "application/vnd.github+json", 15);
        int code = c.getResponseCode();
        if (code != 200) throw new Http(code);
        try (InputStream in = c.getInputStream()) {
            Object perms = JsonParse.object(new String(in.readAllBytes(), StandardCharsets.UTF_8)).get("permissions");
            return perms instanceof Map && Boolean.TRUE.equals(((Map<?, ?>) perms).get("push"));
        }
    }

    static final class Http extends IOException {
        final int code;

        Http(int code) {
            super("HTTP Error " + code);
            this.code = code;
        }
    }

    private Store() {
    }
}
