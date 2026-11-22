package net.svrnet.bridge;

import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

/**
 * A small JSON reader (no libraries): objects as LinkedHashMap (keys in their order), arrays as ArrayList,
 * whole numbers as Long, other numbers as Double, strings, booleans and null. Json.write writes them back.
 */
final class JsonParse {
    private final String s;
    private int i;

    private JsonParse(String s) {
        this.s = s;
    }

    static Object parse(String text) {
        JsonParse p = new JsonParse(text);
        p.ws();
        Object v = p.value();
        p.ws();
        if (p.i != p.s.length()) throw p.error("trailing text");
        return v;
    }

    @SuppressWarnings("unchecked")
    static Map<String, Object> object(String text) {
        Object v = parse(text);
        if (!(v instanceof Map)) throw new IllegalArgumentException("JSON: not an object");
        return (Map<String, Object>) v;
    }

    private IllegalArgumentException error(String what) {
        return new IllegalArgumentException("JSON: " + what + " at " + i);
    }

    private void ws() {
        while (i < s.length() && Character.isWhitespace(s.charAt(i))) i++;
    }

    private Object value() {
        if (i >= s.length()) throw error("unexpected end");
        char c = s.charAt(i);
        switch (c) {
            case '{': return obj();
            case '[': return arr();
            case '"': return str();
            case 't': return lit("true", Boolean.TRUE);
            case 'f': return lit("false", Boolean.FALSE);
            case 'n': return lit("null", null);
            default:
                if (c == '-' || (c >= '0' && c <= '9')) return num();
                if (c == 'N' && s.startsWith("NaN", i)) { i += 3; return Double.NaN; }  // Python's json writes NaN
                if (c == 'I' && s.startsWith("Infinity", i)) { i += 8; return Double.POSITIVE_INFINITY; }
                throw error("unexpected '" + c + "'");
        }
    }

    private Object lit(String word, Object v) {
        if (!s.startsWith(word, i)) throw error("expected " + word);
        i += word.length();
        return v;
    }

    private Map<String, Object> obj() {
        Map<String, Object> m = new LinkedHashMap<>();
        i++;
        ws();
        if (i < s.length() && s.charAt(i) == '}') { i++; return m; }
        while (true) {
            ws();
            String k = str();
            ws();
            if (i >= s.length() || s.charAt(i) != ':') throw error("expected ':'");
            i++;
            ws();
            m.put(k, value());
            ws();
            if (i >= s.length()) throw error("unexpected end");
            char c = s.charAt(i++);
            if (c == '}') return m;
            if (c != ',') throw error("expected ',' or '}'");
        }
    }

    private List<Object> arr() {
        List<Object> l = new ArrayList<>();
        i++;
        ws();
        if (i < s.length() && s.charAt(i) == ']') { i++; return l; }
        while (true) {
            ws();
            l.add(value());
            ws();
            if (i >= s.length()) throw error("unexpected end");
            char c = s.charAt(i++);
            if (c == ']') return l;
            if (c != ',') throw error("expected ',' or ']'");
        }
    }

    private String str() {
        if (i >= s.length() || s.charAt(i) != '"') throw error("expected a string");
        i++;
        StringBuilder b = new StringBuilder();
        while (i < s.length()) {
            char c = s.charAt(i++);
            if (c == '"') return b.toString();
            if (c != '\\') { b.append(c); continue; }
            if (i >= s.length()) break;
            char e = s.charAt(i++);
            switch (e) {
                case '"': b.append('"'); break;
                case '\\': b.append('\\'); break;
                case '/': b.append('/'); break;
                case 'b': b.append('\b'); break;
                case 'f': b.append('\f'); break;
                case 'n': b.append('\n'); break;
                case 'r': b.append('\r'); break;
                case 't': b.append('\t'); break;
                case 'u':
                    if (i + 4 > s.length()) throw error("bad \\u escape");
                    b.append((char) Integer.parseInt(s.substring(i, i + 4), 16));
                    i += 4;
                    break;
                default: throw error("bad escape");
            }
        }
        throw error("unterminated string");
    }

    private Object num() {
        int start = i;
        if (s.charAt(i) == '-') {
            i++;
            if (s.startsWith("Infinity", i)) { i += 8; return Double.NEGATIVE_INFINITY; }
        }
        boolean real = false;
        while (i < s.length()) {
            char c = s.charAt(i);
            if (c >= '0' && c <= '9') i++;
            else if (c == '.' || c == 'e' || c == 'E' || c == '+' || c == '-') { real = true; i++; }
            else break;
        }
        String t = s.substring(start, i);
        if (!real) {
            try {
                return Long.parseLong(t);
            } catch (NumberFormatException ignored) {
                // too long for a long: as a double
            }
        }
        return Double.parseDouble(t);
    }
}
