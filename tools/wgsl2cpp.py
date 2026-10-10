"""Lowers the WGSL kernels to C++, so that every backend runs the same source.

  python tools/wgsl2cpp.py <kernel dir> <namespace> <out.hpp>

Every <name>.wgsl in the directory except header.wgsl is a compute kernel; a line
"// #include header.wgsl" pulls in the shared declarations. Each kernel becomes
`template <class F> struct <Name>` (F = float: the emulated GPU; F = double: the reference):
bindings are pointers, uniforms members, functions member functions, and

  void workgroup(uvec3 wid)

runs one whole workgroup, as the GPU does. `static const char* wgsl()` returns the WGSL itself
(with the header) for WebGPU.

Lowering. A kernel without barriers runs its invocations one after the other. A kernel with
barriers (workgroupBarrier, storageBarrier, workgroupUniformLoad) is lowered the way GPU compilers
lower to CPUs: every function that reaches a barrier is cut at its barriers, and each stretch of
statements between two barriers becomes a loop over the workgroup's lanes (L). Control flow around
barriers is uniform, as WGSL requires: it runs once for the workgroup, evaluated for lane 0.
Variables declared between barriers live on per lane (name_w[L]), as do private variables
(name_p[L]); such variables need an explicit type. Workgroup variables are plain members, zeroed
at the start of every workgroup, as in WGSL.

Only the subset of WGSL the kernels use is accepted (scalars, vec3<u32> builtins, vec4<f32>,
arrays, let/var/const, if/for/loop/break/continue/return, the builtins below); anything else is an
error.
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
| (?P<op>->|&&|\|\||==|!=|<=|>=|\+=|-=|\*=|/=|%=|\+\+|--|[-+*/%<>=!&|^~(){}\[\];:,.@])
""", re.X)

SCALAR = {"f32": "F", "u32": "std::uint32_t", "i32": "std::int32_t", "bool": "bool"}
MATH = {"abs", "exp", "exp2", "log", "log2", "sqrt", "ceil", "floor", "pow", "round", "trunc", "min", "max"}
BARRIERS = {"workgroupBarrier", "storageBarrier"}
UNIFORM_LOAD = "workgroupUniformLoad"
UNSUPPORTED = {"atomicAdd", "atomicLoad", "atomicStore", "textureLoad", "switch", "while", "continuing"}
LANE_BUILTINS = {"local_invocation_index", "local_invocation_id", "global_invocation_id"}


class Error(Exception):
    pass


def tokenize(text):
    out, pos = [], 0
    while pos < len(text):
        m = TOKEN.match(text, pos)
        if not m:
            raise Error(f"unexpected character {text[pos]!r} at {pos}")
        if m.lastgroup not in ("ws", "comment"):
            out.append(m.group())
        pos = m.end()
    return out


def match(toks, i, open_, close):
    """Index of the token closing toks[i] (== open_)."""
    depth = 0
    for j in range(i, len(toks)):
        if toks[j] == open_:
            depth += 1
        elif toks[j] == close:
            depth -= 1
            if depth == 0:
                return j
    raise Error(f"unbalanced {open_}")


def split_type(toks):
    """Splits the tokens of a type from what follows (= ; , ) or { at depth 0)."""
    depth, out = 0, []
    for k, tok in enumerate(toks):
        if tok == "<":
            depth += 1
        elif tok == ">":
            depth -= 1
        elif depth == 0 and tok in ("=", ";", ",", ")", "{"):
            return out, toks[k:]
        out.append(tok)
    return out, []


def ctype(toks, structs):
    """A WGSL type as (C++ element type, array suffix)."""
    s = "".join(toks)
    if s in SCALAR:
        return SCALAR[s], ""
    if s == "vec3<u32>":
        return "uvec3", ""
    m = re.fullmatch(r"vec4<(\w+)>", s)
    if m:
        return f"vec4<{SCALAR[m.group(1)]}>", ""
    m = re.fullmatch(r"array<(.+),(\w+)>", s)
    if m:
        inner, suffix = ctype(tokenize(m.group(1)), structs)
        return inner, f"[{m.group(2)}]{suffix}"
    if s in structs:
        return f"{s}<F>", ""
    raise Error(f"unsupported type {s}")


def is_float(tok):
    return re.fullmatch(r"(?:\d+\.\d*|\.\d+)(?:[eE][+-]?\d+)?[fh]?|\d+[eE][+-]?\d+[fh]?|\d+[fh]", tok) is not None


# ---------------------------------------------------------------------------------------------
# Statements

class Stmt:
    def __init__(self, kind, **kw):
        self.kind, self.toks, self.cond, self.init, self.step, self.body, self.els = kind, [], [], [], [], [], None
        self.__dict__.update(kw)

    def tokens(self):
        """Every token of the statement (for scans)."""
        out = list(self.toks) + list(self.init) + list(self.cond) + list(self.step)
        for s in self.body:
            out += s.tokens()
        if self.els is not None:
            out += self.els.tokens()
        return out


def parse_block(toks):
    out, i = [], 0
    while i < len(toks):
        st, i = parse_stmt(toks, i)
        out.append(st)
    return out


def parse_stmt(toks, i):
    t = toks[i]
    if t in UNSUPPORTED:
        raise Error(f"{t} is not supported")
    if t == "{":
        j = match(toks, i, "{", "}")
        return Stmt("block", body=parse_block(toks[i + 1:j])), j + 1
    if t == "if":
        if toks[i + 1] != "(":
            raise Error("if conditions must be parenthesised")
        j = match(toks, i + 1, "(", ")")
        if toks[j + 1] != "{":
            raise Error("if conditions must be wholly parenthesised")
        m = match(toks, j + 1, "{", "}")
        st = Stmt("if", cond=toks[i + 2:j], body=parse_block(toks[j + 2:m]))
        n = m + 1
        if n < len(toks) and toks[n] == "else":
            if toks[n + 1] == "if":
                st.els, n = parse_stmt(toks, n + 1)
            else:
                m2 = match(toks, n + 1, "{", "}")
                st.els, n = Stmt("block", body=parse_block(toks[n + 2:m2])), m2 + 1
        return st, n
    if t == "for":
        j = match(toks, i + 1, "(", ")")
        parts, depth, cur = [], 0, []
        for tok in toks[i + 2:j]:
            if tok in ("(", "["):
                depth += 1
            elif tok in (")", "]"):
                depth -= 1
            if tok == ";" and depth == 0:
                parts.append(cur)
                cur = []
            else:
                cur.append(tok)
        parts.append(cur)
        m = match(toks, j + 1, "{", "}")
        return Stmt("for", init=parts[0], cond=parts[1], step=parts[2], body=parse_block(toks[j + 2:m])), m + 1
    if t == "loop":
        m = match(toks, i + 1, "{", "}")
        return Stmt("loop", body=parse_block(toks[i + 2:m])), m + 1
    depth, j = 0, i
    while True:
        tok = toks[j]
        if tok in ("(", "["):
            depth += 1
        elif tok in (")", "]"):
            depth -= 1
        elif tok == ";" and depth == 0:
            break
        j += 1
    return Stmt("simple", toks=toks[i:j + 1]), j + 1


# ---------------------------------------------------------------------------------------------
# Expressions

def substitute(toks, names):
    out = []
    for k, tok in enumerate(toks):
        if tok in names and (k == 0 or toks[k - 1] != ".") and (k + 1 >= len(toks) or toks[k + 1] != ":"):
            out += tokenize(names[tok])
        else:
            out.append(tok)
    return out


def expr(toks, structs):
    """Translates a statement or an expression token by token."""
    out, k = [], 0
    while k < len(toks):
        tok, nxt = toks[k], toks[k + 1] if k + 1 < len(toks) else None
        if tok in UNSUPPORTED:
            raise Error(f"{tok} is not supported")
        if tok in ("let", "var"):
            if nxt == "<":
                raise Error("address-spaced var inside a function")
            colon = toks[k + 2] if k + 2 < len(toks) else None
            k += 2
            if colon == ":":
                ty, rest = split_type(toks[k + 1:])
                k += 1 + len(ty)
                base, suffix = ctype(ty, structs)
                decl = ("const " if tok == "let" else "") + base + " " + nxt + suffix
                if not rest or rest[0] != "=":
                    decl += "{}"
                out.append(decl)
            else:
                out.append(("const auto " if tok == "let" else "auto ") + nxt)
            continue
        if tok == "vec4" and nxt == "<":
            out.append(f"vec4<{SCALAR[toks[k + 2]]}>")
            k += 4
            continue
        if tok == "select" and nxt == "(":
            out.append("wsel")
        elif tok in SCALAR and nxt == "(":
            out.append(SCALAR[tok])
        elif tok in MATH and nxt == "(":
            out.append("std::" + tok)
        elif is_float(tok):
            out.append(f"F({tok.rstrip('fh')})")
        elif re.fullmatch(r"\d+i", tok):
            out.append(tok[:-1])
        else:
            out.append(tok)
        k += 1
    return out


def render(toks):
    """Joins tokens into one line of readable C++."""
    cur, prev, unary = "", None, False
    for tok in toks:
        word = prev is not None and re.search(r"[\w\)\]>]$", prev) is not None and prev != "return"
        if not cur or unary or tok in (")", "]", ";", ",", ".", "++", "--") or prev in ("(", "[", ".", "!"):
            sep = ""
        elif tok in ("(", "["):
            sep = "" if word and prev not in ("if", "for", "while", "auto") else " "
        else:
            sep = " "
        unary = tok in ("-", "!", "~") and not word
        cur += sep + tok
        prev = tok
    return cur


# ---------------------------------------------------------------------------------------------
# Emission

def calls(toks):
    return {toks[k] for k in range(len(toks) - 1) if toks[k + 1] == "("}


def escapes(stmts, in_loop=False):
    """Whether the statements return, or break / continue out of a loop they are not inside."""
    for s in stmts:
        if s.kind == "simple":
            if s.toks[0] == "return" or (not in_loop and s.toks[0] in ("break", "continue")):
                return True
        elif s.kind in ("for", "loop"):
            if escapes(s.body, True):
                return True
        elif s.kind == "if":
            if escapes(s.body, in_loop) or (s.els is not None and escapes([s.els], in_loop)):
                return True
        elif s.kind == "block" and escapes(s.body, in_loop):
            return True
    return False


class Emitter:
    def __init__(self, structs, barrier_fns, lanes, private_vars, lane_names=()):
        self.structs, self.barrier_fns, self.lanes, self.lane_names = structs, barrier_fns, lanes, set(lane_names)
        self.arrays = set()
        self.priv = {v: f"{v}_p[L]" for v in private_vars} if barrier_fns else {}

    def is_wg(self, s):
        toks = s.tokens()
        if any(t in BARRIERS or t == UNIFORM_LOAD for t in toks) or calls(toks) & self.barrier_fns:
            return True
        return escapes([s])

    def tr(self, toks, names):
        return render(expr(substitute(toks, names), self.structs))

    # Per-invocation statements.
    def plain(self, stmts, names, ind):
        names = dict(names)
        out = []
        for s in stmts:
            out += self.plain_stmt(s, names, ind)
        return out

    def plain_stmt(self, s, names, ind, prefix=""):
        pad = "  " * ind
        if s.kind == "simple":
            if s.toks[0] in BARRIERS or UNIFORM_LOAD in s.toks:
                raise Error("barrier outside uniform control flow")
            line = pad + prefix + self.tr(s.toks, names)
            if s.toks[0] in ("let", "var"):
                names.pop(s.toks[1], None)  # the declaration shadows any per-lane name
            return [line]
        if s.kind == "block":
            return [pad + prefix + "{"] + self.plain(s.body, names, ind + 1) + [pad + "}"]
        if s.kind == "if":
            out = [pad + prefix + "if (" + self.tr(s.cond, names) + ") {"] + self.plain(s.body, names, ind + 1) + [pad + "}"]
            if s.els is not None:
                if s.els.kind == "if":
                    out += self.plain_stmt(s.els, names, ind, "else ")
                else:
                    out += [pad + "else {"] + self.plain(s.els.body, names, ind + 1) + [pad + "}"]
            return out
        if s.kind == "for":
            inner = dict(names)
            if s.init and s.init[0] in ("let", "var"):
                inner.pop(s.init[1], None)
            head = "for (" + self.tr(s.init, inner) + "; " + self.tr(s.cond, inner) + "; " + self.tr(s.step, inner) + ") {"
            return [pad + prefix + head] + self.plain(s.body, inner, ind + 1) + [pad + "}"]
        if s.kind == "loop":
            return [pad + prefix + "for (;;) {"] + self.plain(s.body, names, ind + 1) + [pad + "}"]
        raise Error(s.kind)

    # Workgroup-level (lowered) statements.
    def lowered(self, stmts, names, ind, prelude):
        names = dict(names)
        out, run = [], []
        for s in stmts:
            if self.is_wg(s):
                out += self.flush(run, names, ind, prelude)
                run = []
                out += self.lowered_stmt(s, names, ind, prelude)
            else:
                run.append(s)
        out += self.flush(run, names, ind, prelude)
        return out

    def lane0(self, s):
        """Whether the statement only runs on lane 0: if (<lane index> == 0u) { ... } without else."""
        c = s.cond
        return s.kind == "if" and s.els is None and len(c) == 3 and c[0] in self.lane_names and c[1] == "==" and c[2] in ("0u", "0")

    def flush(self, run, names, ind, prelude):
        if not run:
            return []
        pad = "  " * ind
        if all(self.lane0(s) for s in run):
            # Only lane 0 does anything: run it alone (the other lanes would skip every statement).
            body = []
            for s in run:
                body += self.plain(s.body, names, ind + 1)
            return [f"{pad}{{", f"{pad}  L = 0;"] + [pad + "  " + x for x in prelude] + body + [f"{pad}}}"]
        decls, body, cached = [], [], []
        used = {t for st in run for t in st.tokens()}
        local = dict(names)
        # Per-lane variables the stretch uses live in locals while a lane runs it (registers, not
        # memory the compiler must assume aliased), stored back at its end.
        for v in [v for v, r in names.items() if r == f"{v}_w[L]" and v in used and v not in self.arrays]:
            body.append(f"{pad}  auto {v} = {v}_w[L];")
            local.pop(v)
            cached.append(v)
        for st in run:
            if st.kind == "simple" and st.toks[0] in ("let", "var"):
                vname = st.toks[1]
                if st.toks[2] != ":":
                    raise Error(f"{vname}: a variable declared between barriers needs an explicit type")
                ty, rest = split_type(st.toks[3:])
                base, suffix = ctype(ty, self.structs)
                init = rest[1:-1] if rest and rest[0] == "=" else None
                decls.append(f"{pad}[[maybe_unused]] {base} {vname}_w[{self.lanes}]{suffix}{{}};")
                names[vname] = f"{vname}_w[L]"
                if suffix:
                    self.arrays.add(vname)
                    local[vname] = f"{vname}_w[L]"
                    body.append(f"{pad}  for (auto& x_ : {vname}_w[L]) x_ = {{}};")
                    if init is not None:
                        raise Error(f"{vname}: per-lane arrays cannot be initialised")
                    continue
                value = self.tr(init, local) if init is not None else "{}"
                body.append(f"{pad}  {base} {vname} = {value};")
                local.pop(vname, None)
                cached.append(vname)
            else:
                body += self.plain_stmt(st, local, ind + 1)
        body += [f"{pad}  {v}_w[L] = {v};" for v in cached]
        return decls + [f"{pad}for (L = 0; L < kLanes; ++L) {{"] + [pad + "  " + x for x in prelude] + body + [f"{pad}}}", f"{pad}L = 0;"]

    def lowered_stmt(self, s, names, ind, prelude, prefix=""):
        pad = "  " * ind
        if s.kind == "simple":
            toks = s.toks
            if toks[0] in BARRIERS:
                return [f"{pad}// {toks[0]}"]
            if UNIFORM_LOAD in toks:
                k = toks.index(UNIFORM_LOAD)
                j = match(toks, k + 1, "(", ")")
                if toks[k + 2] != "&":
                    raise Error("workgroupUniformLoad(&variable)")
                toks = toks[:k] + ["("] + toks[k + 3:j] + [")"] + toks[j + 1:]
            if toks[0] in names and names[toks[0]].endswith("_w[L]") and toks[1] == "=":
                # A per-lane variable given a workgroup-wide value: every lane's copy gets it.
                value = self.tr(toks[2:-1], names)
                return [f"{pad}{prefix}{{", f"{pad}  const auto v_ = {value};", f"{pad}  for (L = 0; L < kLanes; ++L) {names[toks[0]]} = v_;",
                        f"{pad}  L = 0;", f"{pad}}}"]
            line = pad + prefix + self.tr(toks, names)
            if toks[0] in ("let", "var"):
                names.pop(toks[1], None)  # uniform: one value for the workgroup
            return [line]
        if s.kind == "block":
            return [pad + prefix + "{"] + self.lowered(s.body, names, ind + 1, prelude) + [pad + "}"]
        if s.kind == "if":
            out = [pad + prefix + "if (" + self.tr(s.cond, names) + ") {"] + self.lowered(s.body, names, ind + 1, prelude) + [pad + "}"]
            if s.els is not None:
                if s.els.kind == "if":
                    out += self.lowered_stmt(s.els, names, ind, prelude, "else ")
                else:
                    out += [pad + "else {"] + self.lowered(s.els.body, names, ind + 1, prelude) + [pad + "}"]
            return out
        if s.kind == "for":
            inner = dict(names)
            if s.init and s.init[0] in ("let", "var"):
                inner.pop(s.init[1], None)
            head = "for (" + self.tr(s.init, inner) + "; " + self.tr(s.cond, inner) + "; " + self.tr(s.step, inner) + ") {"
            return [pad + prefix + head] + self.lowered(s.body, inner, ind + 1, prelude) + [pad + "}"]
        if s.kind == "loop":
            return [pad + prefix + "for (;;) {"] + self.lowered(s.body, names, ind + 1, prelude) + [pad + "}"]
        raise Error(s.kind)


def lane_vec(wg, var):
    x, y = wg[0], wg[1]
    return f"uvec3{{{var} % {x}u, ({var} / {x}u) % {y}u, {var} / {x * y}u}}"


def translate(source, structs, struct_defs, name):
    # "// wgsl2cpp: override NAME = value": the value the CPU translation gives an override constant.
    cpu = {m.group(1): m.group(2) for m in re.finditer(r"//\s*wgsl2cpp:\s*override\s+(\w+)\s*=\s*(\w+)", source)}
    ints = {}
    toks = tokenize(source)
    members, fns, private_vars, workgroup_vars = [], [], [], []
    entry, wg = None, [1, 1, 1]
    i = 0
    while i < len(toks):
        attrs = {}
        while toks[i] == "@":
            aname = toks[i + 1]
            i += 2
            if i < len(toks) and toks[i] == "(":
                j = match(toks, i, "(", ")")
                attrs[aname] = [t for t in toks[i + 1:j] if t != ","]
                i = j + 1
            else:
                attrs[aname] = []
        kw = toks[i]
        if kw == "struct":
            sname = toks[i + 1]
            j = match(toks, i + 2, "{", "}")
            body, i = toks[i + 3:j], j + 1
            if i < len(toks) and toks[i] == ";":
                i += 1
            fields, q = [], 0
            while q < len(body):
                fname = body[q]
                ty, rest = split_type(body[q + 2:])
                q += 2 + len(ty) + (1 if rest and rest[0] == "," else 0)
                m = re.fullmatch(r"array<vec4<(\w+)>,(\d+)>", "".join(ty))
                if m:
                    fields.append(f"{SCALAR[m.group(1)]} {fname}[{m.group(2)}][4]{{}};")
                else:
                    base, suffix = ctype(ty, structs)
                    fields.append(f"{base} {fname}{suffix}{{}};")
            structs.add(sname)
            struct_defs.setdefault(sname, fields)
        elif kw == "var":
            i += 1
            space = []
            if toks[i] == "<":
                j = match(toks, i, "<", ">")
                space, i = [t for t in toks[i + 1:j] if t != ","], j + 1
            vname = toks[i]
            j = toks.index(";", i)
            ty = toks[i + 2:j]
            i = j + 1
            binding = "".join(attrs.get("binding", []))
            if space and space[0] == "storage":
                m = re.fullmatch(r"array<(\w+)>", "".join(ty))
                if not m:
                    raise Error(f"storage {vname}: {''.join(ty)}")
                members.append(f"{SCALAR[m.group(1)]}* {vname} = nullptr;  // binding {binding}")
            elif space and space[0] == "uniform":
                members.append(f"{ctype(ty, structs)[0]} {vname}{{}};  // binding {binding}")
            elif space and space[0] == "private":
                private_vars.append(vname)
                members.append(("private", vname) + ctype(ty, structs))
            elif space and space[0] == "workgroup":
                base, suffix = ctype(ty, structs)
                workgroup_vars.append((vname, bool(suffix)))
                members.append(f"{base} {vname}{suffix}{{}};  // workgroup")
            else:
                raise Error(f"var<{','.join(space)}> {vname} is not supported")
        elif kw in ("const", "override"):
            cname = toks[i + 1]
            j = toks.index("=", i)
            e = toks.index(";", j)
            value = toks[j + 1:e]
            if kw == "override" and cname in cpu:
                value = tokenize(cpu[cname])
            if len(value) == 1 and re.fullmatch(r"\d+u?", value[0]):
                ints[cname] = int(value[0].rstrip("u"))
            members.append(f"static constexpr {ctype(toks[i + 3:j], structs)[0]} {cname} = {render(expr(value, structs))};")
            i = e + 1
        elif kw == "fn":
            fname = toks[i + 1]
            j = match(toks, i + 2, "(", ")")
            params = toks[i + 3:j]
            i = j + 1
            ret = "void"
            if toks[i] == "->":
                b = toks.index("{", i)
                ret = ctype(toks[i + 1:b], structs)[0]
                i = b
            e = match(toks, i, "{", "}")
            body = parse_block(toks[i + 1:e])
            i = e + 1
            plist, q = [], 0
            while q < len(params):
                builtin = None
                while params[q] == "@":
                    if params[q + 1] != "builtin":
                        raise Error(f"parameter attribute {params[q + 1]}")
                    builtin = params[q + 3]
                    q = match(params, q + 2, "(", ")") + 1
                pname = params[q]
                ty, rest = split_type(params[q + 2:])
                q += 2 + len(ty) + (1 if rest and rest[0] == "," else 0)
                plist.append((pname, ctype(ty, structs)[0], builtin))
            fns.append((fname, plist, ret, body))
            if "compute" in attrs:
                entry = fname
                size = [ints[x] if x in ints else int(x.rstrip("u")) for x in attrs["workgroup_size"]]
                wg = size + [1] * (3 - len(size))
        else:
            raise Error(f"unexpected {kw!r}")
    if entry is None:
        raise Error(f"{name}: no @compute entry point")

    # Functions that reach a barrier, directly or through calls.
    def all_tokens(body):
        return [t for s in body for t in s.tokens()]
    barrier_fns = {f for f, _, _, body in fns if any(t in BARRIERS or t == UNIFORM_LOAD for t in all_tokens(body))}
    changed = True
    while changed:
        changed = False
        for f, _, _, body in fns:
            if f not in barrier_fns and calls(all_tokens(body)) & barrier_fns:
                barrier_fns.add(f)
                changed = True
    lanes = wg[0] * wg[1] * wg[2]
    # Names holding the lane index: the entry's local_invocation_index, and private variables the
    # entry sets to it ("LID = lane;").
    _, eparams, _, ebody = next(f for f in fns if f[0] == entry)
    lane_names = {p for p, _, b in eparams if b == "local_invocation_index"}
    for st in ebody:
        if st.kind == "simple" and len(st.toks) == 4 and st.toks[0] in private_vars and st.toks[1] == "=" and st.toks[2] in lane_names:
            lane_names.add(st.toks[0])
    em = Emitter(structs, barrier_fns, lanes, private_vars, lane_names)

    out = ["template <class F>", f"struct {name} {{",
           f"  static constexpr std::uint32_t kWorkgroup[3] = {{{wg[0]}u, {wg[1]}u, {wg[2]}u}};",
           f"  static constexpr std::uint32_t kLanes = {lanes}u;",
           "  std::uint32_t L = 0;  // the lane being run"]
    for m in members:
        if isinstance(m, tuple):
            _, vname, base, suffix = m
            out.append(f"  {base} {vname}_p[{lanes}]{suffix}{{}};" if barrier_fns else f"  {base} {vname}{suffix}{{}};")
        else:
            out.append("  " + m)
    for fname, plist, ret, body in fns:
        if fname == entry:
            continue
        head = f"{ret} {fname}({', '.join(f'{t} {p}' for p, t, _ in plist)}) {{"
        lines = em.lowered(body, em.priv, 2, []) if fname in barrier_fns else em.plain(body, em.priv, 2)
        out += ["  " + head] + lines + ["  }"]

    # The entry point: one workgroup.
    fname, plist, ret, body = next(f for f in fns if f[0] == entry)
    value = {
        "workgroup_id": "wid_",
        "local_invocation_index": "L",
        "local_invocation_id": "lid_",
        "global_invocation_id": f"uvec3{{wid_.x * {wg[0]}u + lid_.x, wid_.y * {wg[1]}u + lid_.y, wid_.z * {wg[2]}u + lid_.z}}",
    }
    for p, _, b in plist:
        if b not in value:
            raise Error(f"builtin {b} is not supported")
    reset = [f"    for (auto& x_ : {v}) x_ = {{}};" if arr else f"    {v} = {{}};" for v, arr in workgroup_vars]
    if entry in barrier_fns:
        prelude = [f"const uvec3 lid_ = {lane_vec(wg, 'L')};", "(void)lid_;"]
        prelude += [f"const {t} {p} = {value[b]};  (void){p};" for p, t, b in plist if b in LANE_BUILTINS]
        uniform = [f"    const uvec3 {p} = wid_;" for p, _, b in plist if b == "workgroup_id"]
        privs = [f"    for (auto& x_ : {v}_p) x_ = {{}};" for v in private_vars]
        out += ["  void workgroup(uvec3 wid_) {"] + reset + privs + uniform + ["    L = 0;"]
        out += em.lowered(body, em.priv, 2, prelude) + ["  }"]
    else:
        head = f"void {fname}({', '.join(f'{t} {p}' for p, t, _ in plist)}) {{"
        out += ["  " + head] + em.plain(body, em.priv, 2) + ["  }"]
        args = ", ".join(value[b] for _, _, b in plist)
        out += ["  void workgroup(uvec3 wid_) {"] + reset + [
            "    for (L = 0; L < kLanes; ++L) {",
            f"      const uvec3 lid_ = {lane_vec(wg, 'L')};",
            "      (void)lid_;"] + [f"      {v} = {{}};" for v in private_vars] + [f"      {fname}({args});", "    }", "    L = 0;", "  }"]
    out.append(f'  static const char* wgsl() {{ return R"wgsl({source})wgsl"; }}')
    out.append("};")
    return out


def camel(stem):
    return "".join(p[:1].upper() + p[1:] for p in re.split(r"[_\-]", stem))


PRELUDE = r"""#ifndef OFM_WGSL_PRELUDE
#define OFM_WGSL_PRELUDE
namespace ofm::wgsl {
struct uvec3 { std::uint32_t x = 0, y = 0, z = 0; };
template <class T> T wsel(T f, T t, bool c) { return c ? t : f; }
template <class T>
struct vec4 {
  T x{}, y{}, z{}, w{};
  vec4() = default;
  vec4(T a, T b, T c, T d) : x(a), y(b), z(c), w(d) {}
  explicit vec4(T s) : x(s), y(s), z(s), w(s) {}
  friend vec4 operator+(vec4 a, vec4 b) { return {a.x + b.x, a.y + b.y, a.z + b.z, a.w + b.w}; }
  friend vec4 operator-(vec4 a, vec4 b) { return {a.x - b.x, a.y - b.y, a.z - b.z, a.w - b.w}; }
  friend vec4 operator*(vec4 a, vec4 b) { return {a.x * b.x, a.y * b.y, a.z * b.z, a.w * b.w}; }
  friend vec4 operator*(vec4 a, T s) { return {a.x * s, a.y * s, a.z * s, a.w * s}; }
  friend vec4 operator*(T s, vec4 a) { return a * s; }
  friend vec4 operator-(vec4 a) { return {-a.x, -a.y, -a.z, -a.w}; }
  vec4& operator+=(vec4 b) { return *this = *this + b; }
};
template <class T> T dot(vec4<T> a, vec4<T> b) { return a.x * b.x + a.y * b.y + a.z * b.z + a.w * b.w; }
}  // namespace ofm::wgsl
#endif
"""


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
    out = ["// Generated by tools/wgsl2cpp.py from kernels/" + os.path.basename(os.path.normpath(src_dir)) + "/*.wgsl. Do not edit.",
           "#pragma once", "#include <algorithm>", "#include <cmath>", "#include <cstdint>", "", PRELUDE,
           f"namespace {namespace} {{", "using ofm::wgsl::uvec3;", "using ofm::wgsl::vec4;", "using ofm::wgsl::wsel;", ""]
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
