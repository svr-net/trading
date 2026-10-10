"""Translates the WGSL kernels into C++, so that every backend runs the same source.

  python tools/wgsl2cpp.py <kernel dir> <namespace> <out.hpp>

Every <name>.wgsl in the directory except header.wgsl is a compute kernel; a line
"// #include header.wgsl" pulls in the shared declarations. Each kernel becomes
`template <class F> struct <Name>`: its bindings are pointers, its uniform and private variables
members, its functions member functions, and its entry point `void main(uvec3 id)`, run once per
invocation by the host. F is the floating-point type: float emulates the GPU, double is the
reference. `static const char* wgsl()` returns the WGSL itself (with the header) for WebGPU.

Only the subset of WGSL the kernels use is accepted (scalars, u32/i32/f32/bool, arrays as
bindings, vec4<u32> in the uniform, let/var/const, if/for/loop/break/continue/return, the
builtins below); anything else, including barriers and workgroup memory, is an error.
"""
import os
import re
import sys

TOKEN = re.compile(r"""
  (?P<ws>\s+)
| (?P<comment>//[^\n]*)
| (?P<float>(?:\d+\.\d*|\.\d+)(?:[eE][+-]?\d+)?[fh]?|\d+[eE][+-]?\d+[fh]?|\d+[fh])
| (?P<int>0[xX][0-9a-fA-F]+[ui]?|\d+[ui]?)
| (?P<ident>[A-Za-z_]\w*)
| (?P<op>->|&&|\|\||==|!=|<=|>=|<<|>>|[-+*/%<>=!&|^~(){}\[\];:,.@])
""", re.X)

SCALAR = {"f32": "F", "u32": "std::uint32_t", "i32": "std::int32_t", "bool": "bool"}
MATH = {"abs", "exp", "exp2", "log", "log2", "sqrt", "ceil", "floor", "pow", "round", "trunc"}
MINMAX = {"min", "max"}
UNSUPPORTED = {"workgroupBarrier", "storageBarrier", "atomicAdd", "textureLoad", "switch"}


class Error(Exception):
    pass


def tokenize(text):
    out, pos = [], 0
    while pos < len(text):
        m = TOKEN.match(text, pos)
        if not m:
            raise Error(f"unexpected character {text[pos]!r} at {pos}")
        out.append((m.lastgroup, m.group()))
        pos = m.end()
    return out


class Stream:
    def __init__(self, tokens):
        self.t = [t for t in tokens if t[0] not in ("ws", "comment")]
        self.i = 0

    def peek(self, k=0):
        j = self.i + k
        return self.t[j][1] if j < len(self.t) else None

    def next(self):
        tok = self.t[self.i][1]
        self.i += 1
        return tok

    def expect(self, s):
        tok = self.next()
        if tok != s:
            raise Error(f"expected {s!r}, got {tok!r}")
        return tok

    def until_matching(self, open_, close):
        """Tokens up to the matching close (consumed), after the opening one."""
        self.expect(open_)
        depth, out = 1, []
        while True:
            tok = self.next()
            if tok == open_:
                depth += 1
            elif tok == close:
                depth -= 1
                if depth == 0:
                    return out
            out.append(tok)

    def done(self):
        return self.i >= len(self.t)


def ctype(toks, structs):
    """A WGSL type (token list) as C++."""
    s = "".join(toks)
    if s in SCALAR:
        return SCALAR[s]
    if s == "vec3<u32>":
        return "uvec3"
    if s in structs:
        return f"{s}<F>"
    raise Error(f"unsupported type {s}")


def split_type(toks):
    """Splits the tokens of a type from what follows (stops at = ; , ) or { at depth 0)."""
    depth, out = 0, []
    for k, tok in enumerate(toks):
        if tok == "<":
            depth += 1
        elif tok == ">":
            depth -= 1
        elif tok == ">>" and depth >= 2:
            depth -= 2
        elif depth == 0 and tok in ("=", ";", ",", ")", "{"):
            return out, toks[k:]
        out.append(tok)
    return out, []


def is_float(tok):
    return re.fullmatch(r"(?:\d+\.\d*|\.\d+)(?:[eE][+-]?\d+)?[fh]?|\d+[eE][+-]?\d+[fh]?|\d+[fh]", tok) is not None


def expr(toks, structs):
    """Translates statements / expressions token by token."""
    out, k = [], 0
    while k < len(toks):
        tok, nxt = toks[k], toks[k + 1] if k + 1 < len(toks) else None
        if tok in UNSUPPORTED:
            raise Error(f"{tok} is not supported")
        if tok in ("let", "var"):
            if nxt == "<":
                raise Error("address-spaced var inside a function")
            name = toks[k + 2] if k + 2 < len(toks) else None
            k += 2
            if name == ":":
                ty, rest = split_type(toks[k + 1:])
                k += 1 + len(ty)
                decl = ("const " if tok == "let" else "") + ctype(ty, structs) + " " + nxt
                if rest and rest[0] == ";":
                    decl += "{}"
                out.append(decl)
            else:
                out.append(("const auto " if tok == "let" else "auto ") + nxt)
            continue
        if tok == "loop":
            out.append("for (;;)")
        elif tok == "select" and nxt == "(":
            out.append("wsel")
        elif tok in SCALAR and nxt == "(":
            out.append(SCALAR[tok])
        elif tok in MATH and nxt == "(":
            out.append("std::" + tok)
        elif tok in MINMAX and nxt == "(":
            out.append("std::" + tok)
        elif is_float(tok):
            out.append(f"F({tok.rstrip('fh')})")
        elif re.fullmatch(r"\d+i", tok):
            out.append(tok[:-1])
        else:
            out.append(tok)
        k += 1
    return out


def render(toks, indent):
    """Joins tokens into readable C++ (one statement per line)."""
    lines, cur, depth, paren, prev, unary = [], "", indent, 0, None, False
    for tok in toks:
        if tok == "{":
            lines.append("  " * depth + cur.strip() + " {")
            cur, depth, prev = "", depth + 1, tok
            continue
        if tok == "}":
            if cur.strip():
                lines.append("  " * depth + cur.strip())
            depth -= 1
            lines.append("  " * depth + "}")
            cur, prev = "", tok
            continue
        word = prev is not None and re.search(r"[\w\)\]]$", prev) is not None
        if not cur or unary or tok in (")", "]", ";", ",", ".") or prev in ("(", "[", ".", "!"):
            sep = ""
        elif tok in ("(", "["):
            sep = "" if word and prev not in ("if", "for", "while", "return", "auto") else " "
        else:
            sep = " "
        unary = tok in ("-", "!", "~") and not word
        if tok == "(":
            paren += 1
        elif tok == ")":
            paren -= 1
        cur += sep + tok
        prev = tok
        if tok == ";" and paren == 0:
            lines.append("  " * depth + cur.strip())
            cur = ""
    if cur.strip():
        lines.append("  " * depth + cur.strip())
    return lines


def attributes(s):
    attrs = {}
    while s.peek() == "@":
        s.next()
        name = s.next()
        args = s.until_matching("(", ")") if s.peek() == "(" else []
        attrs[name] = args
    return attrs


def translate(source, structs, struct_defs, name):
    """One kernel (header included) as a C++ struct template."""
    s = Stream(tokenize(source))
    members, functions, bindings, wg = [], [], [], None
    while not s.done():
        attrs = attributes(s)
        kw = s.next()
        if kw == "struct":
            sname = s.next()
            body = s.until_matching("{", "}")
            if s.peek() == ";":
                s.next()
            fields, k = [], 0
            while k < len(body):
                fname = body[k]
                ty, rest = split_type(body[k + 2:])
                k += 2 + len(ty) + (1 if rest and rest[0] == "," else 0)
                tys = "".join(ty)
                m = re.fullmatch(r"array<vec4<(\w+)>,(\d+)>", tys)
                if m:
                    fields.append(f"{SCALAR[m.group(1)]} {fname}[{m.group(2)}][4]{{}};")
                else:
                    fields.append(f"{ctype(ty, structs)} {fname}{{}};")
            structs.add(sname)
            struct_defs.setdefault(sname, fields)
        elif kw == "var":
            space = s.until_matching("<", ">") if s.peek() == "<" else []
            vname = s.next()
            s.expect(":")
            ty = []
            while s.peek() != ";":
                ty.append(s.next())
            s.expect(";")
            tys = "".join(ty)
            if space and space[0] == "storage":
                m = re.fullmatch(r"array<(\w+)>", tys)
                if not m:
                    raise Error(f"storage {vname}: {tys}")
                members.append(f"{SCALAR[m.group(1)]}* {vname} = nullptr;  // binding {''.join(attrs.get('binding', []))}")
                bindings.append(vname)
            elif space and space[0] == "uniform":
                members.append(f"{ctype(ty, structs)} {vname}{{}};  // binding {''.join(attrs.get('binding', []))}")
            elif space and space[0] == "private":
                members.append(f"{ctype(ty, structs)} {vname}{{}};")
            else:
                raise Error(f"var<{''.join(space)}> {vname} is not supported")
        elif kw == "const":
            cname = s.next()
            s.expect(":")
            ty = []
            while s.peek() != "=":
                ty.append(s.next())
            s.expect("=")
            val = []
            while s.peek() != ";":
                val.append(s.next())
            s.expect(";")
            members.append(f"static constexpr {ctype(ty, structs)} {cname} = {' '.join(expr(val, structs))};")
        elif kw == "fn":
            fname = s.next()
            params = s.until_matching("(", ")")
            ret = "void"
            if s.peek() == "->":
                s.next()
                ty = []
                while s.peek() != "{":
                    ty.append(s.next())
                ret = ctype(ty, structs)
            body = s.until_matching("{", "}")
            plist, k = [], 0
            while k < len(params):
                pattrs = []
                while params[k] == "@":
                    j = params.index(")", k)
                    pattrs.append(params[k + 1])
                    k = j + 1
                pname = params[k]
                ty, rest = split_type(params[k + 2:])
                k += 2 + len(ty) + (1 if rest and rest[0] == "," else 0)
                if pattrs and pattrs != ["builtin"]:
                    raise Error(f"parameter attributes {pattrs}")
                plist.append(f"{ctype(ty, structs)} {pname}")
            if "compute" in attrs:
                if fname != "main" or len(plist) != 1 or not plist[0].startswith("uvec3 "):
                    raise Error("the entry point must be main(@builtin(global_invocation_id) id : vec3<u32>)")
                wg = [int(x) for x in attrs["workgroup_size"] if x != ","]
                wg += [1] * (3 - len(wg))
            head = f"{ret} {fname}({', '.join(plist)})"
            functions.append([head + " {"] + render(expr(body, structs), 1) + ["}"])
        else:
            raise Error(f"unexpected {kw!r}")
    if wg is None:
        raise Error(f"{name}: no @compute entry point")
    out = [f"template <class F>", f"struct {name} {{",
           f"  static constexpr std::uint32_t kWorkgroup[3] = {{{wg[0]}u, {wg[1]}u, {wg[2]}u}};"]
    out += ["  " + m for m in members]
    for f in functions:
        out += ["  " + line for line in f]
    out.append(f'  static const char* wgsl() {{ return R"wgsl({source})wgsl"; }}')
    out.append("};")
    return out


def camel(stem):
    return "".join(p[:1].upper() + p[1:] for p in re.split(r"[_\-]", stem))


def main():
    src_dir, namespace, out_path = sys.argv[1:4]
    names = sorted(f for f in os.listdir(src_dir) if f.endswith(".wgsl") and f != "header.wgsl")
    structs, struct_defs, kernels = set(), {}, []
    for f in names:
        with open(os.path.join(src_dir, f)) as h:
            text = h.read()

        def include(m):
            with open(os.path.join(src_dir, m.group(1))) as inc:
                return inc.read()
        text = re.sub(r"^// #include (\S+)\n", include, text, flags=re.M)
        try:
            kernels.append(translate(text, structs, struct_defs, camel(f[:-5])))
        except Error as e:
            sys.exit(f"{os.path.join(src_dir, f)}: {e}")
    out = ["// Generated by tools/wgsl2cpp.py from " + src_dir.replace("\\", "/").split("kernels/")[-1] + "/*.wgsl. Do not edit.",
           "#pragma once", "#include <algorithm>", "#include <cmath>", "#include <cstdint>", "",
           "#ifndef OFM_WGSL_PRELUDE", "#define OFM_WGSL_PRELUDE",
           "namespace ofm::wgsl {",
           "struct uvec3 { std::uint32_t x = 0, y = 0, z = 0; };",
           "template <class T> T wsel(T f, T t, bool c) { return c ? t : f; }",
           "}  // namespace ofm::wgsl", "#endif", "",
           f"namespace {namespace} {{", "using ofm::wgsl::uvec3;", "using ofm::wgsl::wsel;", ""]
    for sname, fields in struct_defs.items():
        out += ["template <class F>", f"struct {sname} {{"] + ["  " + x for x in fields] + ["};", ""]
    for k in kernels:
        out += k + [""]
    out.append(f"}}  // namespace {namespace}")
    text = "\n".join(out) + "\n"
    try:
        with open(out_path) as h:
            if h.read() == text:
                return
    except FileNotFoundError:
        pass
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w") as h:
        h.write(text)


if __name__ == "__main__":
    main()
