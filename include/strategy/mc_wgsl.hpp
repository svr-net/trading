// The nested Monte Carlo's trade kernel in WGSL, embedded into the WebAssembly engine's webgpu back end
// (backends/webgpu.cpp) as its one source: one invocation per (stop setting, share, path), statement for statement
// mc_gpu.hpp's mcBatchRef<float>. Kept here, beside the reference it mirrors.
#pragma once

namespace strat {

inline const char* const kMcWgsl = R"WGSL(
// The nested Monte Carlo's trade kernel on the GPU (backends/webgpu.cpp embeds it): one invocation per (stop setting,
// share, path), statement for statement include/strategy/mc_gpu.hpp's mcBatchRef<float> (the path's bars from the
// reset's window, then the book's rules on them: the buy under the cap, the stop, the buy-back). Single precision
// (WebGPU has no doubles); its agreement with the engine's double precision is checked by tools/check/mcgpu_check.sh
// and the Lab's end-to-end test web/tests/labgpu.mjs.

struct Params { P: u32, H: u32, S: u32, G: u32, D: u32 }
struct Share {
  last0: f32, sig0: f32, adj: f32, cap: f32, trip: f32,
  lows: array<f32, 5>, highs: array<f32, 5>, nl: u32,
  mark: f32, sar: f32, ep: f32, af: f32, held: u32,
}
struct Spec { stopLoss: f32, rebuy: f32, afStep: f32, afMax: f32, vstop: f32, psarVol: f32, rebuyStop: f32, flags: u32 }

@group(0) @binding(0) var<uniform> prm: Params;
@group(0) @binding(1) var<storage, read> days: array<u32>;
@group(0) @binding(2) var<storage, read> tbl: array<f32>;
@group(0) @binding(3) var<storage, read> shares: array<Share>;
@group(0) @binding(4) var<storage, read> specs: array<Spec>;
@group(0) @binding(5) var<storage, read_write> outRet: array<f32>;
@group(0) @binding(6) var<storage, read_write> outStop: array<u32>;

const NO_DAY: u32 = 0xFFFFFFFFu;

var<private> lows: array<f32, 5>;
var<private> highs: array<f32, 5>;
var<private> nl: u32;

fn lowOf(n: u32) -> f32 {
  var m = 0.0;
  for (var j = nl - min(n, nl); j < nl; j++) { if (m > 0.0) { m = min(m, lows[j]); } else { m = lows[j]; } }
  return m;
}
fn highOf(n: u32) -> f32 {
  var m = 0.0;
  for (var j = nl - min(n, nl); j < nl; j++) { m = max(m, highs[j]); }
  return m;
}

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3u, @builtin(num_workgroups) nwg: vec3u) {
  let id = gid.x + gid.y * nwg.x * 64u;
  let total = prm.G * prm.S * prm.P;
  if (id >= total) { return; }
  let p = id % prm.P;
  let s = (id / prm.P) % prm.S;
  let g = id / (prm.P * prm.S);
  let sh = shares[s];
  let sp = specs[g];
  outRet[id] = 0.0;
  outStop[id] = 0u;
  let last0 = sh.last0;
  if (!(last0 > 0.0)) { return; }
  let psar = (sp.flags & 1u) != 0u;
  let trail = (sp.flags & 2u) != 0u;
  let tripOn = (sp.flags & 4u) != 0u;
  let sig0 = sh.sig0; let cap = sh.cap; let adj = sh.adj;
  for (var k = 0u; k < 5u; k++) { lows[k] = sh.lows[k]; highs[k] = sh.highs[k]; }
  nl = sh.nl;
  let stops = psar || sp.stopLoss > 0.0;
  var inPos = (sh.held & 1u) != 0u;
  var pending = !inPos;
  var stopped = false; var turned = false; var outStopped = false;
  var w = 1.0;
  var mark = select(0.0, sh.mark, inPos);
  var sar = sh.sar; var ep = sh.ep; var af = sh.af;
  var vd = select(0.0, sh.af, inPos);
  var hic = select(0.0, sh.ep / (1.0 - sh.af), inPos && sh.af < 1.0);
  var exitNext = (sh.held & 2u) != 0u;
  var dsar = 0.0; var dep = 0.0; var daf = 0.0; var level = 0.0; var low = 0.0;
  var band = 0.0;
  if (sp.psarVol > 0.0) { band = clamp(sp.psarVol * sig0, 0.02, 0.5); }
  var last = last0;
  for (var k = 0u; k < prm.H; k++) {
    let q = days[p * prm.H + k];
    if (q == NO_DAY) { continue; }
    let at = (s * prm.D + q) * 4u;
    let r0 = tbl[at]; let r1 = tbl[at + 1u]; let r2 = tbl[at + 2u]; let r3 = tbl[at + 3u];
    if (r3 == 0.0) { continue; }
    let base = last * adj;
    let o = base * r0; let hi = base * r1; let lo = base * r2; let c = base * r3;
    let l5 = lowOf(5u);
    if (nl == 5u) {
      for (var j = 0u; j < 4u; j++) { lows[j] = lows[j + 1u]; highs[j] = highs[j + 1u]; }
      nl = 4u;
    }
    lows[nl] = lo; highs[nl] = hi; nl = nl + 1u;
    let l2 = lowOf(2u); let h2 = highOf(2u);
    let prev = last;
    last = c;
    if (sp.vstop > 0.0) {
      if (!inPos && pending && o <= prev * (1.0 + cap)) {
        if (outStopped && tripOn) { w *= 1.0 - sh.trip; }
        inPos = true; pending = false; mark = o; vd = clamp(sp.vstop * sig0, 0.02, 0.5);
        ep = o * (1.0 - vd); sar = o * (1.0 - 1.5 * vd); hic = o; exitNext = false;
      }
      if (inPos) {
        if (exitNext) {
          w *= o / mark; inPos = false; outStopped = true; stopped = true; exitNext = false; level = ep;
        } else if (lo <= sar) {
          w *= min(o, sar) / mark; inPos = false; outStopped = true; stopped = true; level = ep;
        } else {
          w *= c / mark; mark = c;
          if (c < ep) {
            exitNext = true;
          } else {
            hic = max(hic, c); ep = max(ep, hic * (1.0 - vd)); sar = max(sar, ep * (1.0 - 0.5 * vd));
          }
        }
      } else if (stopped && sp.rebuy >= 0.0 && c >= level * (1.0 + sp.rebuy)) {
        pending = true; stopped = false;
      }
      continue;
    }
    // enter(px), inlined at each call: a position opened at px, with its stop
    if (!inPos && pending && o <= prev * (1.0 + cap)) {
      let px = o;
      if (outStopped && tripOn) { w *= 1.0 - sh.trip; }
      inPos = true; pending = false; stopped = false; mark = px;
      if (psar && sp.psarVol > 0.0) { sar = px * (1.0 - band); ep = px; af = sp.afStep; }
      else if (psar) { sar = select(0.95 * px, l5, l5 > 0.0 && l5 < px); ep = px; af = sp.afStep; }
      else { sar = select(0.0, px * (1.0 - sp.stopLoss), sp.stopLoss > 0.0); }
    }
    var intraday = false;
    if (!inPos && stopped && sp.rebuyStop > 0.0) {
      let trig = select(level, low, trail) * (1.0 + sp.rebuyStop);
      var px = 0.0;
      var buy = false;
      if (o >= trig) { px = o; buy = true; }
      else if (hi >= trig) { px = trig; buy = true; intraday = true; }
      else if (lo < low) { low = lo; }
      if (buy) {
        if (outStopped && tripOn) { w *= 1.0 - sh.trip; }
        inPos = true; pending = false; stopped = false; mark = px;
        if (psar && sp.psarVol > 0.0) { sar = px * (1.0 - band); ep = px; af = sp.afStep; }
        else if (psar) { sar = select(0.95 * px, l5, l5 > 0.0 && l5 < px); ep = px; af = sp.afStep; }
        else { sar = select(0.0, px * (1.0 - sp.stopLoss), sp.stopLoss > 0.0); }
      }
    }
    if (inPos) {
      if (stops && sar > 0.0 && lo <= sar) {
        let px = select(min(o, sar), sar, intraday);
        w *= px / mark; inPos = false; outStopped = true; stopped = true; turned = false; level = px; low = min(px, lo);
        if (psar) { dsar = max(ep, px); dep = px; daf = sp.afStep; }
        if (sp.rebuyStop > 0.0 && !intraday && c >= px * (1.0 + sp.rebuyStop)) {
          let bx = px * (1.0 + sp.rebuyStop);
          if (outStopped && tripOn) { w *= 1.0 - sh.trip; }
          inPos = true; pending = false; stopped = false; mark = bx;
          if (psar && sp.psarVol > 0.0) { sar = bx * (1.0 - band); ep = bx; af = sp.afStep; }
          else if (psar) { sar = select(0.95 * bx, l5, l5 > 0.0 && l5 < bx); ep = bx; af = sp.afStep; }
          else { sar = select(0.0, bx * (1.0 - sp.stopLoss), sp.stopLoss > 0.0); }
          w *= c / mark; mark = c;
        }
      } else {
        w *= c / mark; mark = c;
        if (psar) {
          if (hi > ep) { ep = hi; af = min(sp.afMax, af + sp.afStep); }
          var next = sar + af * (ep - sar);
          next = min(next, l2);
          if (sp.psarVol > 0.0) { next = min(next, c * (1.0 - band)); }
          sar = max(sar, next);
        }
      }
    } else if (stopped && sp.rebuy >= 0.0 && sp.rebuyStop <= 0.0) {
      if (psar) {
        if (c > dsar) {
          turned = true;
        } else {
          if (lo < dep) { dep = lo; daf = min(sp.afMax, daf + sp.afStep); }
          dsar = min(dsar, max(dsar + daf * (dep - dsar), h2));
          continue;
        }
      } else {
        turned = c >= level * (1.0 + sp.rebuy);
      }
      if (turned) { pending = true; stopped = false; }
    }
  }
  outRet[id] = w - 1.0;
  outStop[id] = select(0u, 1u, outStopped);
}
)WGSL";

}  // namespace strat
