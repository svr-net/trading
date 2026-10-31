// #include header.wgsl
// The day's trades, fused, first half: every candidate share (a slot) under every entry and exit
// setting (limit depth, stop tail, take tail, SAR weight) bought with its limit and sold at its
// stop-loss, take-profit or the close, net of costs; once per share, however many rankings hold it
// (records.wgsl then adds each setting's day). The fields of P: T the settings per share (t0 u0 u0
// u1), N the slots, K a slot's stride, t0 the limit depths, u0 the tail fractions, u1 the SAR
// weights, u2 1 to sell at the close (no levels). A setting decodes as (limit, stop tail, take tail,
// SAR weight), the last fastest. st [N][K] per slot: open, high, low, close of the trading day, the
// previous close, sigma, the half-spread factor, the SAR (0 none), 1 with 20 days of history, the
// stamp duty on buying it (UK; 0 for AIM shares), then
// the falls at the u0 tail fractions and the rises at them (logs from the open over the last 100
// days). dv [t0] the limit depths (sigmas below the previous close), av [u1] the SAR weights, cost [3]
// the broker's costs (as in costOf; cost[2] unused, the stamp duty is the slot's). ret [N][T]: the trade's return (0 when the limit is not reached,
// or when the stop-loss sits within a spread of the entry: a trade stopped as it fills is skipped).
@group(0) @binding(1) var<storage, read> st : array<f32>;
@group(0) @binding(2) var<storage, read> dv : array<f32>;
@group(0) @binding(3) var<storage, read> av : array<f32>;
@group(0) @binding(4) var<storage, read> cost : array<f32>;
@group(0) @binding(5) var<storage, read_write> ret : array<f32>;

// A filled trade's costs, a share of its value: bought at entry (a share's price in the account's
// currency for the US) and sold r higher.
fn costOf(entry : f32, r : f32, stamp : f32) -> f32 {
  let cap : f32 = cost[1];
  if (cost[0] == 1.0) {
    let sell : f32 = cap * (1.0 + r);
    return (max(1.0, 0.0005 * cap) + 0.00005 * cap + max(1.0, 0.0005 * sell) + 0.00005 * sell) / cap + stamp;
  }
  if (cost[0] == 2.0) {
    let n : f32 = max(1.0, floor(cap / entry));
    let v : f32 = n * entry;
    let sell : f32 = v * (1.0 + r);
    let cb : f32 = min(max(0.35, 0.0035 * n), 0.01 * v) + 0.0005 * n;
    let cs : f32 = min(max(0.35, 0.0035 * n), 0.01 * sell) + 0.0005 * n;
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
    if (o <= lim) { entry = o; } else if (l <= lim / half) { entry = lim; intra = true; }
    if (entry > 0.0) {
      var stop : f32 = 0.0;
      var take : f32 = 1.0e30;
      if (P.u2 == 0u && st[b + 8u] > 0.5) {
        var ls : f32 = st[b + 10u + cs];
        if (sar > 0.0) { ls = (1.0 - av[ca]) * ls + av[ca] * log(min(1.0, sar / entry)); }
        stop = entry * exp(ls);
        take = entry * exp(st[b + 10u + P.u0 + ct]);
      }
      // A stop within a spread of the entry would sell as the trade fills: no trade (cash).
      if (stop * half * half <= entry) {
        var px : f32 = c;
        if (l <= stop) { px = stop / half; } else if (!intra && h >= take * half) { px = take; }
        let gross : f32 = px / entry - 1.0;
        y = gross - costOf(entry, gross, st[b + 9u]);
      }
    }
    ret[i] = y;
  }
}
