package net.svrnet.bridge;

import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

/** Just enough JSON for the bridge: writing maps, lists, strings, numbers and booleans; reading one string field. */
final class Json {
    static String obj(Object... kv) {
        Map<String, Object> m = new LinkedHashMap<>();
        for (int i = 0; i + 1 < kv.length; i += 2) m.put(String.valueOf(kv[i]), kv[i + 1]);
        return write(m);
    }

    static String write(Object v) {
        StringBuilder b = new StringBuilder();
        write(b, v);
        return b.toString();
    }

    private static void write(StringBuilder b, Object v) {
        if (v == null) {
            b.append("null");
        } else if (v instanceof Boolean) {
            b.append(v);
        } else if (v instanceof Number) {
            double d = ((Number) v).doubleValue();
            if (Double.isNaN(d) || Double.isInfinite(d)) b.append("null");
            else if (v instanceof Double || v instanceof Float) b.append(d);
            else b.append(v);
        } else if (v instanceof Map) {
            b.append('{');
            boolean first = true;
            for (Map.Entry<?, ?> e : ((Map<?, ?>) v).entrySet()) {
                if (!first) b.append(',');
                first = false;
                str(b, String.valueOf(e.getKey()));
                b.append(':');
                write(b, e.getValue());
            }
            b.append('}');
        } else if (v instanceof List) {
            b.append('[');
            boolean first = true;
            for (Object o : (List<?>) v) {
                if (!first) b.append(',');
                first = false;
                write(b, o);
            }
            b.append(']');
        } else {
            str(b, String.valueOf(v));
        }
    }

    private static void str(StringBuilder b, String s) {
        b.append('"');
        for (int i = 0; i < s.length(); i++) {
            char c = s.charAt(i);
            switch (c) {
                case '"': b.append("\\\""); break;
                case '\\': b.append("\\\\"); break;
                case '\n': b.append("\\n"); break;
                case '\r': b.append("\\r"); break;
                case '\t': b.append("\\t"); break;
                default:
                    if (c < 0x20) b.append(String.format("\\u%04x", (int) c));
                    else b.append(c);
            }
        }
        b.append('"');
    }

    private static final Pattern FIELD = Pattern.compile("\"(%s)\"\\s*:\\s*\"((?:[^\"\\\\]|\\\\.)*)\"");

    /** A top-level string field of a small JSON object ("" when absent). */
    static String field(String json, String name) {
        Matcher m = Pattern.compile(String.format(FIELD.pattern(), Pattern.quote(name))).matcher(json == null ? "" : json);
        return m.find() ? m.group(2).replace("\\\"", "\"").replace("\\\\", "\\") : "";
    }

    private Json() {
    }
}
