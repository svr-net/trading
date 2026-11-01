// #include header.wgsl
// The day's trades, fused, first half: every candidate share (a slot) under every entry and exit
// setting (limit depth, stop tail, take tail, SAR weight) bought with its limit and sold at its
// stop-loss, take-profit or the close, net of costs; once per share, however many rankings hold it
// (records.wgsl then adds each setting's day). The fields of P: T the settings per share (t0 u0 u0
// u1), N the slots, K a slot's stride, t0 the limit depths, u0 the tail fractions, u1 the SAR
// weights, u2 1 to sell at the close (no levels), 2 a safety stop (no take-profit; u1 then its recovery stops' windows, not SAR weights). A setting decodes as (limit, stop tail, take tail,
// SAR weight), the last fastest. st [N][K] per slot: open, high, low, close of the trading day, the
// previous close, sigma, the half-spread factor, the SAR (0 none), 1 with 20 days of history, its
// seven cost fields (see costOf), then the falls at the u0 tail fractions and the rises at them
// (logs from the open over the last 100 days), then a safety stop's u1 recovery stops (logs; -50 none). dv [t0] the limit depths (sigmas below the previous
// close), av [u1] the SAR weights. ret [N][T]: the trade's return (0 when the limit is not reached,
// or when the stop-loss sits within a spread of the entry: a trade stopped as it fills is skipped).
@group(0) @binding(1) var<storage, read> st : array<f32>;
@group(0) @binding(2) var<storage, read> dv : array<f32>;
@group(0) @binding(3) var<storage, read> av : array<f32>;
@group(0) @binding(4) var<storage, read_write> ret : array<f32>;

// A filled trade's costs, a share of its value, from the slot's cost fields (b + 9: kind, the
// amount a trade, the commission's rate, its minimum, a fee's rate, the tax on buying and the tax on
// both sides; kind 1 a rate with a minimum an order, 2 the US's commission a share): bought at entry
// (a share's price in the trade's currency) and sold r higher.
fn costOf(entry : f32, r : f32, b : u32) -> f32 {
  let kind : f32 = st[b + 9u];
  let cap : f32 = st[b + 10u];
  if (kind == 1.0) {
    let sell : f32 = cap * (1.0 + r);
    let pct : f32 = st[b + 11u];
    let lo : f32 = st[b + 12u];
    let fee : f32 = st[b + 13u];
    return (max(lo, pct * cap) + fee * cap + max(lo, pct * sell) + fee * sell) / cap + st[b + 14u] + st[b + 15u] * (2.0 + r);
  }
  if (kind == 2.0) {  // a rate a share (b + 11), at least a minimum (b + 12), at most 1%, + a fee a share (b + 13)
    let n : f32 = max(1.0, floor(cap / entry));
    let v : f32 = n * entry;
    let sell : f32 = v * (1.0 + r);
    let cb : f32 = min(max(st[b + 12u], st[b + 11u] * n), 0.01 * v) + st[b + 13u] * n;
    let cs : f32 = min(max(st[b + 12u], st[b + 11u] * n), 0.01 * sell) + st[b + 13u] * n;
    return (cb + cs + 0.0000278 * sell + min(0.000166 * n, 8.3)) / v;
  }
  return 0.0;
}

@compute @workgroup_size(64)
fn main(@builtin(workgroup_id) wid : vec3<u32>, @builtin(local_invocation_index) lane : u32) {
  let i : u32 = wid.x * 64u + lane;
  if (i < P.N * P.T) {
    let slot : u32 = i / P.T;
    let li : u32 = i % P.T;
    let ca : u32 = li % P.u1;
    var r : u32 = li / P.u1;
    let ct : u32 = r % P.u0;
    r = r / P.u0;
    let cs : u32 = r % P.u0;
    let ce : u32 = r / P.u0;
    let b : u32 = slot * P.K;
    let o : f32 = st[b];
    let h : f32 = st[b + 1u];
    let l : f32 = st[b + 2u];
    let c : f32 = st[b + 3u];
    let half : f32 = st[b + 6u];
    let sar : f32 = st[b + 7u];
    let lim : f32 = st[b + 4u] * exp(-dv[ce] * st[b + 5u]);
    var entry : f32 = 0.0;
    var intra : bool = false;
    var y : f32 = 0.0;
    if (o <= lim) { entry = o; if (st[b + 8u] > 1.5) { entry = lim; } } else if (l <= lim / half) { entry = lim; intra = true; }
    if (entry > 0.0) {
      // The levels as the orders set them before the open: from the anchor, the limit or the previous
      // close when the limit is above it (the fill's price is not known when they are placed).
      let anchor : f32 = min(lim, st[b + 4u]);
      var fs : f32 = 0.0;
      var take : f32 = 1.0e30;
      if (P.u2 != 1u && st[b + 8u] > 0.5) {
        var ls : f32 = st[b + 16u + cs];
        if (P.u2 == 2u) { ls = max(ls, st[b + 16u + 2u * P.u0 + ca]); }  // a safety stop: no deeper than the share's recovery stop
        else if (sar > 0.0) { ls = (1.0 - av[ca]) * ls + av[ca] * log(min(1.0, sar / anchor)); }
        fs = exp(ls);
        take = anchor * exp(st[b + 16u + P.u0 + ct]);
      }
      let stop : f32 = anchor * fs;
      // A stop within a spread of the anchor: no order placed (cash), as the orders skip it.
      if (fs * half * half <= 1.0) {
        var px : f32 = c;
        if (!intra && o <= stop) { px = o / half; }          // opened through the stop: sold at the open
        else if (l <= stop) { px = stop / half; }
        else if (!intra && h >= take * half) { px = max(take, o); }  // a take-profit below the open fills at the open
        let gross : f32 = px / entry - 1.0;
        y = gross - costOf(entry, gross, b);
      }
    }
    ret[i] = y;
  }
}
