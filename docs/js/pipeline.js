// Clean-up, calibration, 3D volume, dye detection and the dye report.
import {
  dilate, feret, gaussianBlur, mainTissue, median, percentiles, tick, tissueMask,
} from "./imgproc.js";
import { warp, warpMask } from "./register.js";
import { anatomicalPosition, describe } from "./orientation.js";

// ---------------------------------------------------------------- dye channel guess
const DYE_WORDS = ["dextran", "red", "tomato", "texas", "rhodamine", "594", "568", "555", "cy3", "dye", "mcherry"];
export function guessDyeChannel(names, sampleChans) {
  const byName = names.findIndex((n) => DYE_WORDS.some((w) => n.toLowerCase().includes(w)));
  if (byName >= 0) return byName;
  if (sampleChans && sampleChans.length > 1) {
    const scores = sampleChans.map((x) => {
      const [md, p90, p999] = percentiles(x, [50, 90, 99.9], 60000);
      return (p999 - md) / (Math.abs(p90 - md) + 1e-3 * (Math.abs(p999 - md) + 1));
    });
    const sorted = [...scores].sort((a, b) => b - a);
    if (sorted[0] > 2 * sorted[1]) return scores.indexOf(sorted[0]);
  }
  return 0;
}

// ---------------------------------------------------------------- clean-up
/** Background subtraction, light smoothing, stack-wide 0..1 scaling. Returns {proc, masks}. */
export async function preprocess(stack, opts, progress = () => {}) {
  const n = stack.size, S = stack.slices.length, nC = stack.names.length;
  const masks = [];
  for (let i = 0; i < S; i++) {
    masks.push(tissueMask(stack.slices[i].chans, n, n, opts.maskChannels, opts.sensitivity));
    progress((40 * (i + 1)) / S, `Finding tissue in slice ${i + 1}`);
    await tick();
  }
  const cleaned = [];
  for (let i = 0; i < S; i++) {
    const outside = masks[i].map((v) => (v ? 0 : 1));
    cleaned.push(stack.slices[i].chans.map((x) => {
      const bg = median(x, outside);
      const y = new Float32Array(x.length);
      for (let k = 0; k < x.length; k++) y[k] = Math.max(0, x[k] - bg);
      return opts.denoise > 0 ? gaussianBlur(y, n, n, opts.denoise) : y;
    }));
    progress(40 + (40 * (i + 1)) / S, `Cleaning slice ${i + 1}`);
    await tick();
  }
  // One brightness scale per channel for the whole stack (keeps real differences).
  const proc = cleaned.map(() => []);
  for (let c = 0; c < nC; c++) {
    const parts = cleaned.map((ch) => percentiles(ch[c], [1, 99.97], 40000));
    const lo = Math.min(...parts.map((p) => p[0]));
    const hiSorted = parts.map((p) => p[1]).sort((a, b) => a - b);
    const hi = hiSorted[Math.floor(0.9 * (hiSorted.length - 1))] || 1;
    for (let i = 0; i < S; i++) {
      const x = cleaned[i][c], y = new Float32Array(x.length);
      for (let k = 0; k < x.length; k++) y[k] = Math.min(1, Math.max(0, (x[k] - lo) / (hi - lo + 1e-6)));
      proc[i].push(y);
    }
  }
  progress(100, "Clean-up done");
  return { proc, masks };
}

// ---------------------------------------------------------------- calibration
export function tissueExtents(masks, n, axis) {
  return masks.map((m) => { const [s, l] = feret(mainTissue(m, n, n), n, n); return axis === "height" ? s : l; });
}
export function referenceSlice(ext) {
  const pos = ext.filter((e) => e > 0).sort((a, b) => a - b);
  const med = pos.length ? pos[pos.length >> 1] : 0;
  let best = 0;
  ext.forEach((e, i) => { if (e <= 1.25 * med && e > ext[best]) best = i; });
  if (ext[best] > 1.25 * med) best = ext.indexOf(Math.max(...ext));
  return best;
}

// ---------------------------------------------------------------- 3D volume
export async function buildVolume(stack, proc, masks, tfs, cal, progress = () => {}) {
  const n = stack.size, S = proc.length, nC = stack.names.length;
  const warped = [], wm = [];
  for (let i = 0; i < S; i++) {
    warped.push(proc[i].map((ch) => warp(ch, n, tfs[i], 0)));
    wm.push(warpMask(masks[i], n, tfs[i]));
    progress((90 * (i + 1)) / S, `Placing slice ${i + 1}`);
    await tick();
  }
  // Crop to the tissue.
  let x0 = n, x1 = -1, y0 = n, y1 = -1;
  for (const m of wm) for (let y = 0; y < n; y++) for (let x = 0; x < n; x++) if (m[y * n + x]) {
    if (x < x0) x0 = x; if (x > x1) x1 = x; if (y < y0) y0 = y; if (y > y1) y1 = y;
  }
  if (x1 < 0) { x0 = 0; y0 = 0; x1 = n - 1; y1 = n - 1; }
  const pad = 8;
  x0 = Math.max(0, x0 - pad); y0 = Math.max(0, y0 - pad); x1 = Math.min(n - 1, x1 + pad); y1 = Math.min(n - 1, y1 + pad);
  const W = x1 - x0 + 1, H = y1 - y0 + 1;
  const chans = Array.from({ length: nC }, () => new Uint8Array(S * H * W));
  const mask = new Uint8Array(S * H * W);
  for (let z = 0; z < S; z++) for (let y = 0; y < H; y++) for (let x = 0; x < W; x++) {
    const src = (y + y0) * n + (x + x0), dst = (z * H + y) * W + x;
    mask[dst] = wm[z][src];
    for (let c = 0; c < nC; c++) chans[c][dst] = Math.round(warped[z][c][src] * 255);
  }
  progress(100, "3D model ready");
  return { Z: S, H, W, chans, mask, xy: cal.xy, z: cal.z, cal, names: [...stack.names] };
}

export function brainVolumeMm3(vol) {
  let c = 0; for (const v of vol.mask) c += v;
  return (c * vol.xy * vol.xy * vol.z) / 1e9;
}

// ---------------------------------------------------------------- dye detection
function autoLevel(vals, k) {
  if (!vals.length) return 255;
  const s = Float32Array.from(vals).sort();
  const q = (p) => s[Math.min(s.length - 1, Math.round(p * (s.length - 1)))];
  const med = q(0.5);
  const dev = Float32Array.from(s, (v) => Math.abs(v - med)).sort();
  const spread = Math.max(1, dev[dev.length >> 1] * 1.4826);
  return Math.min(250, Math.max(3, Math.round(Math.max(med + k * spread, 0.05 * q(0.9999)))));
}

/**
 * settings: { channel, mode: 'auto'|'core'|'manual', level, insideOnly, dyeOnly }
 */
export function detectDye(vol, s) {
  const { Z, H, W } = vol, plane = H * W;
  const k = s.dyeOnly ? 4 : 8;
  const sm = new Float32Array(Z * plane);
  const region = new Uint8Array(Z * plane);
  const sliceThr = new Float32Array(Z);
  const allVals = [];
  for (let z = 0; z < Z; z++) {
    const src = new Float32Array(plane);
    for (let i = 0; i < plane; i++) src[i] = vol.chans[s.channel][z * plane + i];
    const b = gaussianBlur(src, W, H, 1);
    sm.set(b, z * plane);
    const reg = s.insideOnly ? dilate(vol.mask.subarray(z * plane, (z + 1) * plane), W, H, 2) : new Uint8Array(plane).fill(1);
    region.set(reg, z * plane);
    const vals = [];
    const step = Math.max(1, Math.floor(plane / 150000));
    for (let i = 0; i < plane; i += step) if (reg[i]) vals.push(b[i]);
    sliceThr[z] = vals.length > 500 ? autoLevel(vals, k) : 255;
    for (let i = 0; i < vals.length; i += 4) allVals.push(vals[i]);
  }
  const globalAuto = autoLevel(allVals, k);
  const sortedAll = Float32Array.from(allVals).sort();
  const top = sortedAll.length ? sortedAll[Math.round(0.9999 * (sortedAll.length - 1))] : 255;
  const core = Math.max(globalAuto, Math.min(250, Math.max(3, Math.round(0.5 * top))));
  let thr;
  const used = new Float32Array(Z);
  if (s.mode === "core") { thr = core; used.fill(core); }
  else if (s.mode === "manual") { thr = s.level; used.fill(s.level); }
  else {
    // Each slice against its own background, so one bright slice can't hide the rest.
    used.set(sliceThr);
    const valid = Array.from(sliceThr).filter((v) => v < 255).sort((a, b) => a - b);
    thr = valid.length ? Math.round(valid[valid.length >> 1]) : globalAuto;
  }
  let mask = new Uint8Array(Z * plane);
  for (let z = 0; z < Z; z++) for (let i = 0; i < plane; i++) {
    const j = z * plane + i;
    if (region[j] && sm[j] >= used[z]) mask[j] = 1;
  }
  // 3D connected pieces; drop specks smaller than 0.5% of the biggest piece.
  const { labels, sizes } = label3d(mask, Z, H, W);
  const maxSize = Math.max(0, ...sizes.slice(1));
  const minSize = Math.max(4, 0.005 * maxSize);
  const keep = sizes.map((sz, i) => i > 0 && sz >= minSize);
  const voxUm3 = vol.xy * vol.xy * vol.z;
  const dep = new Map();
  for (let j = 0; j < mask.length; j++) {
    if (!mask[j]) continue;
    const L = labels[j];
    if (!keep[L]) { mask[j] = 0; continue; }
    let d = dep.get(L);
    if (!d) { d = { n: 0, w: 0, x: 0, y: 0, z: 0, peak: 0 }; dep.set(L, d); }
    const z = Math.floor(j / plane), r = j - z * plane, y = Math.floor(r / W), x = r - y * W, v = sm[j];
    d.n++; d.w += v; d.x += v * x; d.y += v * y; d.z += v * z; d.peak = Math.max(d.peak, vol.chans[s.channel][j]);
  }
  const deposits = [...dep.values()].map((d) => ({
    volumeMm3: (d.n * voxUm3) / 1e9,
    centreUm: [(d.x / d.w) * vol.xy, (d.y / d.w) * vol.xy, (d.z / d.w) * vol.z],
    peak: d.peak, n: d.n,
  })).sort((a, b) => b.volumeMm3 - a.volumeMm3);
  return { settings: { ...s }, threshold: thr, auto: globalAuto, core, sliceThr: used, mask, smoothed: sm, deposits, found: deposits.length > 0 };
}

function label3d(m, Z, H, W) {
  const labels = new Int32Array(m.length), sizes = [0], plane = H * W;
  const stack = new Int32Array(m.length);
  let n = 0;
  for (let s = 0; s < m.length; s++) {
    if (!m[s] || labels[s]) continue;
    n++; labels[s] = n; let top = 0, size = 0; stack[top++] = s;
    while (top) {
      const i = stack[--top]; size++;
      const z = Math.floor(i / plane), r = i - z * plane, y = Math.floor(r / W), x = r - y * W;
      for (let dz = -1; dz <= 1; dz++) {
        const zz = z + dz; if (zz < 0 || zz >= Z) continue;
        for (let dy = -1; dy <= 1; dy++) {
          const yy = y + dy; if (yy < 0 || yy >= H) continue;
          for (let dx = -1; dx <= 1; dx++) {
            const xx = x + dx; if (xx < 0 || xx >= W) continue;
            const j = (zz * H + yy) * W + xx;
            if (m[j] && !labels[j]) { labels[j] = n; stack[top++] = j; }
          }
        }
      }
    }
    sizes.push(size);
  }
  return { labels, sizes };
}

// ---------------------------------------------------------------- report
export function dyeReport(vol, res) {
  const { Z, H, W } = vol, plane = H * W;
  const brainMm3 = brainVolumeMm3(vol);
  const rep = { found: res.found, threshold: res.threshold, auto: res.auto, core: res.core, deposits: res.deposits, brainMm3 };
  if (!res.found) return rep;
  let cnt = 0, sw = 0, sx = 0, sy = 0, sz = 0;
  let dx0 = W, dx1 = -1, dy0 = H, dy1 = -1, dz0 = Z, dz1 = -1;
  const sliceCount = new Float64Array(Z), sliceSignal = new Float64Array(Z);
  for (let j = 0; j < res.mask.length; j++) {
    if (!res.mask[j]) continue;
    const z = Math.floor(j / plane), r = j - z * plane, y = Math.floor(r / W), x = r - y * W;
    const v = vol.chans[res.settings.channel][j];
    cnt++; sw += v; sx += v * x; sy += v * y; sz += v * z;
    sliceCount[z]++; sliceSignal[z] += v;
    if (x < dx0) dx0 = x; if (x > dx1) dx1 = x; if (y < dy0) dy0 = y; if (y > dy1) dy1 = y; if (z < dz0) dz0 = z; if (z > dz1) dz1 = z;
  }
  let bx0 = W, bx1 = -1, by0 = H, by1 = -1, bz0 = Z, bz1 = -1;
  for (let j = 0; j < vol.mask.length; j++) {
    if (!vol.mask[j]) continue;
    const z = Math.floor(j / plane), r = j - z * plane, y = Math.floor(r / W), x = r - y * W;
    if (x < bx0) bx0 = x; if (x > bx1) bx1 = x; if (y < by0) by0 = y; if (y > by1) by1 = y; if (z < bz0) bz0 = z; if (z > bz1) bz1 = z;
  }
  const cx = (sx / sw) * vol.xy, cy = (sy / sw) * vol.xy, czi = sz / sw, cz = czi * vol.z;
  const dyeMm3 = (cnt * vol.xy * vol.xy * vol.z) / 1e9;
  Object.assign(rep, {
    dyeMm3, pctBrain: (100 * dyeMm3) / brainMm3,
    centreUm: [cx, cy, cz],
    centreFromEdges: [cx - bx0 * vol.xy, cy - by0 * vol.xy, cz - bz0 * vol.z + vol.z / 2],
    brainSize: [(bx1 - bx0) * vol.xy, (by1 - by0) * vol.xy, (bz1 - bz0 + 1) * vol.z],
    extent: [(dx1 - dx0 + 1) * vol.xy, (dy1 - dy0 + 1) * vol.xy, (dz1 - dz0 + 1) * vol.z],
    firstSlice: dz0 + 1, lastSlice: dz1 + 1, centreSlice: Math.round(czi) + 1,
    mostSlice: sliceSignal.indexOf(Math.max(...sliceSignal)) + 1,
    sliceAreasMm2: Array.from(sliceCount, (c) => (c * vol.xy * vol.xy) / 1e6),
  });
  return rep;
}

export function dyeCsv(vol, res, rep, orient, meta) {
  const L = [];
  const row = (...a) => L.push(a.map((v) => (typeof v === "string" && /[",]/.test(v) ? `"${v.replace(/"/g, '""')}"` : v)).join(","));
  row("Red dextran dye - location report (BrainStack 3D web)");
  row("Slicing", describe(orient));
  row("Pixel size XY (um)", vol.xy.toFixed(4));
  row("Slice spacing Z (um)", vol.z.toFixed(2));
  row("Calibration", meta.calibration);
  row("Dye channel", vol.names[res.settings.channel]);
  row("Detection", res.settings.mode, "level used (median of slices)", res.threshold);
  row("Brain volume (mm3)", rep.brainMm3.toFixed(4));
  if (!rep.found) { row("No dye found at this level"); return L.join("\n"); }
  row("Dye volume (mm3)", rep.dyeMm3.toFixed(5));
  row("Dye as % of brain volume", rep.pctBrain.toFixed(3));
  row("Separate deposits", rep.deposits.length);
  const an = anatomicalPosition(rep, orient);
  row("");
  row("Dye centre - anatomical", "um", "% of brain");
  row("Behind the front of the brain", an.fromFront.toFixed(1), an.fromFrontPct.toFixed(1));
  row("Below the top of the brain", an.fromTop.toFixed(1), an.fromTopPct.toFixed(1));
  row(`${an.side} of the midline`, an.midline.toFixed(1), "");
  row("Brain length / height / width (um)", an.length.toFixed(1), an.height.toFixed(1), an.width.toFixed(1));
  row("");
  row("Dye appears in slices", `${rep.firstSlice} to ${rep.lastSlice}`, "centre slice", rep.centreSlice, "slice with the most dye", rep.mostSlice);
  row("Dye spread (um) image X / image Y / slices", ...rep.extent.map((v) => v.toFixed(1)));
  row("");
  row("Deposit", "Volume (mm3)", "Centre X (um)", "Centre Y (um)", "Centre Z (um)", "Peak (0-255)");
  rep.deposits.forEach((d, i) => row(i + 1, d.volumeMm3.toFixed(5), ...d.centreUm.map((v) => v.toFixed(1)), d.peak));
  row("");
  row("Slice", "File", "Z (um)", "Dye level used", "Dye area (mm2)");
  rep.sliceAreasMm2.forEach((a, i) => row(i + 1, meta.files[i] || "", (i * vol.z).toFixed(1), Math.round(res.sliceThr[i]), a.toFixed(5)));
  return L.join("\n");
}
