// Rigid slice-to-slice alignment: shift + rotation, optional mirror.
// Transform = { sx, sy, angle, flip } in working pixels, about the canvas centre.
// angle is degrees, clockwise on screen; flip mirrors left-right BEFORE rotating.
import { gaussianBlur, mainTissue, percentiles, resample, tick } from "./imgproc.js";

export const identity = (flip = false) => ({ sx: 0, sy: 0, angle: 0, flip });

/** Warp a square n×n image. nearest=true for masks. */
export function warp(src, n, t, fill = 0, nearest = false) {
  const out = new Float32Array(n * n);
  const c = (n - 1) / 2, a = (t.angle * Math.PI) / 180, cs = Math.cos(a), sn = Math.sin(a);
  for (let y = 0; y < n; y++) {
    const qy = y - c - t.sy;
    for (let x = 0; x < n; x++) {
      const qx = x - c - t.sx;
      let px = cs * qx + sn * qy + c;          // inverse rotation
      const py = -sn * qx + cs * qy + c;
      if (t.flip) px = n - 1 - px;
      let v = fill;
      if (px >= -0.5 && py >= -0.5 && px <= n - 0.5 && py <= n - 0.5) {
        if (nearest) {
          const ix = Math.min(n - 1, Math.max(0, Math.round(px))), iy = Math.min(n - 1, Math.max(0, Math.round(py)));
          v = src[iy * n + ix];
        } else {
          const xx = Math.min(Math.max(px, 0), n - 1), yy = Math.min(Math.max(py, 0), n - 1);
          const x0 = xx | 0, y0 = yy | 0, x1 = Math.min(x0 + 1, n - 1), y1 = Math.min(y0 + 1, n - 1);
          const ax = xx - x0, ay = yy - y0;
          v = (src[y0 * n + x0] * (1 - ax) + src[y0 * n + x1] * ax) * (1 - ay) +
              (src[y1 * n + x0] * (1 - ax) + src[y1 * n + x1] * ax) * ay;
        }
      }
      out[y * n + x] = v;
    }
  }
  return out;
}

export function warpMask(m, n, t) {
  const f = new Float32Array(m.length);
  for (let i = 0; i < m.length; i++) f[i] = m[i];
  const w = warp(f, n, t, 0, true), out = new Uint8Array(m.length);
  for (let i = 0; i < out.length; i++) out[i] = w[i] > 0.5 ? 1 : 0;
  return out;
}

function ncc(a, b) {
  let sa = 0, sb = 0;
  const n = a.length;
  for (let i = 0; i < n; i++) { sa += a[i]; sb += b[i]; }
  const ma = sa / n, mb = sb / n;
  let num = 0, da = 0, db = 0;
  for (let i = 0; i < n; i++) { const x = a[i] - ma, y = b[i] - mb; num += x * y; da += x * x; db += y * y; }
  return num / Math.sqrt(da * db + 1e-12);
}

function centroid(img, n) {
  let lo = Infinity; for (const v of img) if (v < lo) lo = v;
  let s = 0, sx = 0, sy = 0;
  for (let y = 0; y < n; y++) for (let x = 0; x < n; x++) { const v = img[y * n + x] - lo; s += v; sx += v * x; sy += v * y; }
  return s > 0 ? [sx / s, sy / s] : [(n - 1) / 2, (n - 1) / 2];
}

/** Coordinate-descent refinement of (sx, sy, angle) maximising NCC. */
function refine(fixed, moving, n, t0, maxAngle, steps = [2, 2]) {
  let t = { ...t0 }, best = ncc(fixed, warp(moving, n, t));
  let [ps, as] = steps;
  for (let it = 0; it < 60 && (ps > 0.2 || as > 0.2); it++) {
    let improved = false;
    for (const [k, d] of [["sx", ps], ["sx", -ps], ["sy", ps], ["sy", -ps], ["angle", as], ["angle", -as]]) {
      const c = { ...t, [k]: t[k] + d };
      if (maxAngle != null && Math.abs(c.angle) > maxAngle + 3) continue;
      const s = ncc(fixed, warp(moving, n, c));
      if (s > best + 1e-6) { best = s; t = c; improved = true; }
    }
    if (!improved) { ps /= 2; as /= 2; }
  }
  return { t, score: best };
}

/**
 * Find the transform that lays `moving` on `fixed` (both n×n, already small).
 * opts: { maxAngle: number|null, tryMirror: bool, flip: bool, margin }
 */
export function registerPair(fixed, moving, n, opts = {}) {
  const maxAngle = opts.maxAngle ?? null, flip0 = !!opts.flip;
  // When slices are known to be upright, a mirror is the only big ambiguity left, so
  // even a small but clear improvement decides it. With free rotation, mirror+turn can
  // mimic many shapes, so ask for a bigger improvement before accepting a change.
  const margin = opts.margin ?? (maxAngle != null ? 0.008 : 0.04);
  const flips = opts.tryMirror ? [flip0, !flip0] : [flip0];
  const angles = [];
  if (maxAngle != null) for (let a = -maxAngle; a <= maxAngle + 1e-9; a += 3) angles.push(a);
  else for (let a = -180; a < 180; a += 10) angles.push(a);
  if (!angles.includes(0)) angles.push(0);
  const cf = centroid(fixed, n);
  const cands = [];
  for (const flip of flips) {
    const scored = [];
    for (const a of angles) {
      const rot = warp(moving, n, { sx: 0, sy: 0, angle: a, flip });
      const cm = centroid(rot, n);
      const t = { sx: cf[0] - cm[0], sy: cf[1] - cm[1], angle: a, flip };
      scored.push({ t, score: ncc(fixed, warp(moving, n, t)), start: a });
    }
    scored.sort((p, q) => q.score - p.score);
    const short = scored.slice(0, 4);
    if (!short.some((s) => s.start === 0)) short.push(scored.find((s) => s.start === 0));
    for (const s of short) {
      const r = refine(fixed, moving, n, s.t, maxAngle, [2, maxAngle != null ? 1.5 : 5]);
      cands.push({ ...r, start: s.start, flip });
    }
  }
  cands.sort((p, q) => q.score - p.score);
  let best = cands[0];
  // Prefer: as mounted -> upright (small turn, maybe mirrored) -> best fit.
  const asMounted = cands.filter((c) => c.flip === flip0 && c.start === 0).sort((p, q) => q.score - p.score)[0];
  const upright = cands.filter((c) => Math.abs(c.t.angle) <= 45)[0];
  if (asMounted && asMounted.score >= best.score - margin) best = asMounted;
  else if (upright && upright.score >= best.score - margin) best = upright;
  best.cands = cands;
  return best;
}

/** What to align on when there is no separate brain stain: mostly the tissue outline. */
export function outlineSignal(signal, mask, n, weight = 0.6) {
  const main = mainTissue(mask, n, n);
  const f = new Float32Array(n * n);
  for (let i = 0; i < f.length; i++) f[i] = main[i];
  const shape = gaussianBlur(f, n, n, 3);
  let mx = 1e-6; for (const v of shape) if (v > mx) mx = v;
  const hi = Math.max(1e-6, percentiles(signal, [99.5])[0]);
  const out = new Float32Array(n * n);
  for (let i = 0; i < out.length; i++) out[i] = (weight * shape[i]) / mx + (1 - weight) * Math.min(1, Math.max(0, signal[i] / hi));
  return out;
}

const scaleT = (t, k) => ({ ...t, sx: t.sx * k, sy: t.sy * k });

/**
 * Decide which slices are mirrored, using the whole stack at once.
 * For every pair of nearby slices (up to `reach` apart) we measure how much better
 * they match "same way round" than "one mirrored". Then we pick the set of mirror
 * flags that agrees best with all those pairwise votes (exhaustive for small stacks).
 * This is far more robust than deciding slice by slice along a chain, where one
 * wrong call flips everything after it.
 */
export async function decideMirrors(imgSets, n, maxAngle, progress = () => {}, reach = 4) {
  // imgSets: one or more lists of images (e.g. tissue outline AND dye signal); their votes add up.
  if (!Array.isArray(imgSets[0])) imgSets = [imgSets];
  const N = imgSets[0].length, m = Math.min(n, 96);
  const smalls = imgSets.map((set) => set.map((im) => resample(im, n, n, m, m)));
  const votes = [];   // [i, j, weight] weight > 0 means "same orientation"
  const pairs = [];
  for (let i = 0; i < N; i++) for (let j = i + 1; j <= Math.min(N - 1, i + reach); j++) pairs.push([i, j]);
  for (let k = 0; k < pairs.length; k++) {
    const [i, j] = pairs[k];
    let w = 0;
    for (const small of smalls) {
      const same = registerPair(small[i], small[j], m, { maxAngle, tryMirror: false, flip: false }).score;
      const mir = registerPair(small[i], small[j], m, { maxAngle, tryMirror: false, flip: true }).score;
      w += same - mir;
    }
    votes.push([i, j, w / (j - i)]);
    if (k % 3 === 0) { progress((100 * (k + 1)) / pairs.length, "Checking which slices are mirrored…"); await tick(); }
  }
  const ref = N >> 1;
  const scoreOf = (f) => votes.reduce((acc, [i, j, w]) => acc + (f[i] === f[j] ? w : -w), 0);
  let best = null, bestScore = -Infinity;
  if (N <= 20) {
    for (let mask = 0; mask < 1 << (N - 1); mask++) {
      const f = [];
      for (let i = 0, b = 0; i < N; i++) f.push(i === ref ? false : !!((mask >> b++) & 1));
      const sc = scoreOf(f);
      if (sc > bestScore) { bestScore = sc; best = f; }
    }
  } else { // greedy + local flips for big stacks
    best = new Array(N).fill(false);
    for (let it = 0, improved = true; it < 50 && improved; it++) {
      improved = false;
      for (let i = 0; i < N; i++) { if (i === ref) continue; best[i] = !best[i]; const sc = scoreOf(best); if (sc > bestScore + 1e-9) { bestScore = sc; improved = true; } else best[i] = !best[i]; }
    }
  }
  return best;
}

/**
 * Align a whole stack, middle-out. imgs: n×n Float32Arrays.
 * opts: { flips, ref, maxAngle, tryMirror }
 */
export async function alignStack(imgs, n, opts = {}, progress = () => {}, cancelled = () => false) {
  const N = imgs.length, ref = opts.ref ?? (N >> 1);
  let flips = opts.flips || imgs.map(() => false);
  if (opts.tryMirror) {
    // Mirror votes work best on the tissue OUTLINE alone (its front/back asymmetry).
    const dec = await decideMirrors(opts.mirrorImgs || imgs, n, opts.maxAngle ?? null, (v, msg) => progress(v * 0.4, msg), 3);
    flips = dec;   // decided for the whole stack at once (images are un-mirrored here)
  }
  const prog0 = opts.tryMirror ? 40 : 0;
  const m = Math.min(n, 160), m2 = Math.min(n, 320);
  const small = imgs.map((im) => resample(im, n, n, m, m));
  const mid = imgs.map((im) => resample(im, n, n, m2, m2));
  const tfs = new Array(N).fill(null);
  tfs[ref] = identity(flips[ref]);
  const alS = { [ref]: warp(small[ref], m, tfs[ref]) }, alM = { [ref]: warp(mid[ref], m2, tfs[ref]) };
  const order = [];
  for (let i = ref + 1; i < N; i++) order.push([i, i - 1]);
  for (let i = ref - 1; i >= 0; i--) order.push([i, i + 1]);
  const mean = (arrs) => { const o = new Float32Array(arrs[0].length); for (const a of arrs) for (let i = 0; i < o.length; i++) o[i] += a[i] / arrs.length; return o; };
  for (let k = 0; k < order.length; k++) {
    if (cancelled()) break;
    const [i, nb] = order[k], step = nb - i;
    const nbs = [nb, nb + step].filter((j) => alS[j]);
    const best = registerPair(mean(nbs.map((j) => alS[j])), small[i], m,
      { maxAngle: opts.maxAngle ?? null, tryMirror: false, flip: flips[i] });
    // Polish at higher resolution.
    const fine = refine(mean(nbs.map((j) => alM[j])), mid[i], m2, scaleT(best.t, m2 / m), opts.maxAngle ?? null, [1, 0.5]);
    tfs[i] = scaleT(fine.t, n / m2);
    alS[i] = warp(small[i], m, scaleT(tfs[i], m / n));
    alM[i] = warp(mid[i], m2, scaleT(tfs[i], m2 / n));
    progress(prog0 + ((100 - prog0) * (k + 1)) / order.length, `Aligned slice ${i + 1} of ${N}`);
    await tick();
  }
  return tfs.map((t, j) => t || identity(flips[j]));
}

/** Re-align one slice to a fixed (already aligned) neighbour image. */
export function alignOne(fixedAligned, moving, n, opts) {
  const m = Math.min(n, 160), m2 = Math.min(n, 320);
  const best = registerPair(resample(fixedAligned, n, n, m, m), resample(moving, n, n, m, m), m, opts);
  const fine = refine(resample(fixedAligned, n, n, m2, m2), resample(moving, n, n, m2, m2), m2,
    scaleT(best.t, m2 / m), opts.maxAngle ?? null, [1, 0.5]);
  return scaleT(fine.t, n / m2);
}

/** Turn the whole stack by `deg` about the canvas centre, keeping slices in register. */
export function rotateAll(tfs, deg) {
  const a = (deg * Math.PI) / 180, c = Math.cos(a), s = Math.sin(a);
  return tfs.map((t) => ({ sx: c * t.sx - s * t.sy, sy: s * t.sx + c * t.sy, angle: t.angle + deg, flip: t.flip }));
}
