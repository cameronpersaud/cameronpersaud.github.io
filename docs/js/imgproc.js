// Basic 2D image processing on Float32Array images (row-major, width w, height h).
// Everything here runs in the browser; nothing is uploaded anywhere.

export const tick = () => new Promise((r) => setTimeout(r, 0));

export function gaussianBlur(src, w, h, sigma) {
  if (sigma <= 0) return Float32Array.from(src);
  const r = Math.max(1, Math.ceil(sigma * 3));
  const k = new Float32Array(2 * r + 1);
  let sum = 0;
  for (let i = -r; i <= r; i++) { k[i + r] = Math.exp(-(i * i) / (2 * sigma * sigma)); sum += k[i + r]; }
  for (let i = 0; i < k.length; i++) k[i] /= sum;
  const tmp = new Float32Array(w * h), out = new Float32Array(w * h);
  for (let y = 0; y < h; y++) {
    const row = y * w;
    for (let x = 0; x < w; x++) {
      let acc = 0;
      for (let i = -r; i <= r; i++) {
        let xx = x + i; xx = xx < 0 ? 0 : xx >= w ? w - 1 : xx;
        acc += src[row + xx] * k[i + r];
      }
      tmp[row + x] = acc;
    }
  }
  for (let y = 0; y < h; y++) {
    for (let x = 0; x < w; x++) {
      let acc = 0;
      for (let i = -r; i <= r; i++) {
        let yy = y + i; yy = yy < 0 ? 0 : yy >= h ? h - 1 : yy;
        acc += tmp[yy * w + x] * k[i + r];
      }
      out[y * w + x] = acc;
    }
  }
  return out;
}

/** Area-average resample to (nw, nh). */
export function resample(src, w, h, nw, nh) {
  const out = new Float32Array(nw * nh);
  const fx = w / nw, fy = h / nh;
  if (fx <= 1 && fy <= 1) { // upsample / same: bilinear
    for (let y = 0; y < nh; y++) for (let x = 0; x < nw; x++)
      out[y * nw + x] = bilinear(src, w, h, (x + 0.5) * fx - 0.5, (y + 0.5) * fy - 0.5);
    return out;
  }
  for (let y = 0; y < nh; y++) {
    const y0 = Math.floor(y * fy), y1 = Math.max(y0 + 1, Math.min(h, Math.floor((y + 1) * fy)));
    for (let x = 0; x < nw; x++) {
      const x0 = Math.floor(x * fx), x1 = Math.max(x0 + 1, Math.min(w, Math.floor((x + 1) * fx)));
      let acc = 0;
      for (let yy = y0; yy < y1; yy++) { const row = yy * w; for (let xx = x0; xx < x1; xx++) acc += src[row + xx]; }
      out[y * nw + x] = acc / ((y1 - y0) * (x1 - x0));
    }
  }
  return out;
}

export function bilinear(src, w, h, x, y, fill = 0) {
  if (x < -0.5 || y < -0.5 || x > w - 0.5 || y > h - 0.5) return fill;
  x = Math.min(Math.max(x, 0), w - 1); y = Math.min(Math.max(y, 0), h - 1);
  const x0 = Math.floor(x), y0 = Math.floor(y);
  const x1 = Math.min(x0 + 1, w - 1), y1 = Math.min(y0 + 1, h - 1);
  const ax = x - x0, ay = y - y0;
  const a = src[y0 * w + x0], b = src[y0 * w + x1], c = src[y1 * w + x0], d = src[y1 * w + x1];
  return (a * (1 - ax) + b * ax) * (1 - ay) + (c * (1 - ax) + d * ax) * ay;
}

/** Percentiles from a strided sample (fast and accurate enough for display/thresholds). */
export function percentiles(arr, ps, maxSamples = 200000, mask = null) {
  const step = Math.max(1, Math.floor(arr.length / maxSamples));
  const vals = [];
  for (let i = 0; i < arr.length; i += step) if (!mask || mask[i]) vals.push(arr[i]);
  if (!vals.length) return ps.map(() => 0);
  const s = Float32Array.from(vals).sort();
  return ps.map((p) => s[Math.min(s.length - 1, Math.max(0, Math.round((p / 100) * (s.length - 1))))]);
}

export function median(arr, mask = null) { return percentiles(arr, [50], 200000, mask)[0]; }

export function otsu(arr) {
  let lo = Infinity, hi = -Infinity;
  for (let i = 0; i < arr.length; i++) { const v = arr[i]; if (v < lo) lo = v; if (v > hi) hi = v; }
  if (hi - lo < 1e-9) return lo;
  const bins = 256, hist = new Float64Array(bins);
  for (let i = 0; i < arr.length; i++) hist[Math.min(bins - 1, Math.floor(((arr[i] - lo) / (hi - lo)) * bins))]++;
  const total = arr.length;
  let sumAll = 0; for (let i = 0; i < bins; i++) sumAll += i * hist[i];
  let wB = 0, sumB = 0, best = 0, bestT = 0;
  for (let t = 0; t < bins; t++) {
    wB += hist[t]; if (!wB) continue;
    const wF = total - wB; if (!wF) break;
    sumB += t * hist[t];
    const mB = sumB / wB, mF = (sumAll - sumB) / wF;
    const between = wB * wF * (mB - mF) * (mB - mF);
    if (between > best) { best = between; bestT = t; }
  }
  return lo + ((bestT + 0.5) / bins) * (hi - lo);
}

// ---------------------------------------------------------------- binary masks (Uint8Array 0/1)
export function dilate(m, w, h, iters = 1) {
  let cur = m;
  for (let it = 0; it < iters; it++) {
    const out = new Uint8Array(w * h);
    for (let y = 0; y < h; y++) for (let x = 0; x < w; x++) {
      const i = y * w + x;
      if (cur[i] || (x > 0 && cur[i - 1]) || (x < w - 1 && cur[i + 1]) || (y > 0 && cur[i - w]) || (y < h - 1 && cur[i + w])) out[i] = 1;
    }
    cur = out;
  }
  return cur;
}

export function erode(m, w, h, iters = 1) {
  let cur = m;
  for (let it = 0; it < iters; it++) {
    const out = new Uint8Array(w * h);
    for (let y = 0; y < h; y++) for (let x = 0; x < w; x++) {
      const i = y * w + x;
      if (cur[i] && x > 0 && cur[i - 1] && x < w - 1 && cur[i + 1] && y > 0 && cur[i - w] && y < h - 1 && cur[i + w]) out[i] = 1;
    }
    cur = out;
  }
  return cur;
}

export function fillHoles(m, w, h) {
  // Flood the background from the border; anything not reached is a hole.
  const outside = new Uint8Array(w * h), stack = [];
  const push = (i) => { if (!m[i] && !outside[i]) { outside[i] = 1; stack.push(i); } };
  for (let x = 0; x < w; x++) { push(x); push((h - 1) * w + x); }
  for (let y = 0; y < h; y++) { push(y * w); push(y * w + w - 1); }
  while (stack.length) {
    const i = stack.pop(), x = i % w, y = (i / w) | 0;
    if (x > 0) push(i - 1); if (x < w - 1) push(i + 1); if (y > 0) push(i - w); if (y < h - 1) push(i + w);
  }
  const out = new Uint8Array(w * h);
  for (let i = 0; i < out.length; i++) out[i] = outside[i] ? 0 : 1;
  return out;
}

/** 8-connected component labels. Returns {labels:Int32Array, sizes:[0, s1, s2...]} */
export function label2d(m, w, h) {
  const labels = new Int32Array(w * h), sizes = [0];
  let n = 0;
  const stack = [];
  for (let s = 0; s < m.length; s++) {
    if (!m[s] || labels[s]) continue;
    n++; let size = 0; labels[s] = n; stack.push(s);
    while (stack.length) {
      const i = stack.pop(); size++;
      const x = i % w, y = (i / w) | 0;
      for (let dy = -1; dy <= 1; dy++) for (let dx = -1; dx <= 1; dx++) {
        const xx = x + dx, yy = y + dy;
        if (xx < 0 || yy < 0 || xx >= w || yy >= h) continue;
        const j = yy * w + xx;
        if (m[j] && !labels[j]) { labels[j] = n; stack.push(j); }
      }
    }
    sizes.push(size);
  }
  return { labels, sizes };
}

/** Keep components at least `frac` of the biggest one. */
export function keepBig(m, w, h, frac) {
  const { labels, sizes } = label2d(m, w, h);
  if (sizes.length <= 2) return m;
  const max = Math.max(...sizes.slice(1));
  const out = new Uint8Array(w * h);
  for (let i = 0; i < out.length; i++) if (labels[i] && sizes[labels[i]] >= frac * max) out[i] = 1;
  return out;
}

export function borderMedian(img, w, h) {
  const vals = [];
  for (let x = 0; x < w; x += 2) for (const y of [0, 1, 2, h - 3, h - 2, h - 1]) vals.push(img[y * w + x]);
  for (let y = 0; y < h; y += 2) for (const x of [0, 1, 2, w - 3, w - 2, w - 1]) vals.push(img[y * w + x]);
  vals.sort((a, b) => a - b);
  return vals[vals.length >> 1];
}

/** Tissue (bright) vs slide (dark), from the given channels. */
export function tissueMask(chans, w, h, useChannels, sensitivity = 1) {
  const use = useChannels && useChannels.length ? useChannels : chans.map((_, i) => i);
  const comb = new Float32Array(w * h);
  for (const c of use) {
    const x = chans[c];
    const bg = borderMedian(x, w, h);
    const hi = percentiles(x, [99.9])[0];
    const d = hi - bg + 1e-6;
    for (let i = 0; i < x.length; i++) comb[i] += Math.min(1, Math.max(0, (x[i] - bg) / d)) / use.length;
  }
  const sm = gaussianBlur(comb, w, h, Math.max(3, h / 150));
  let lo = Infinity, hi = -Infinity;
  for (const v of sm) { if (v < lo) lo = v; if (v > hi) hi = v; }
  if (hi - lo < 1e-6) return new Uint8Array(w * h);
  const thr = (otsu(sm) * 0.6) / Math.max(sensitivity, 1e-3);
  let m = new Uint8Array(w * h);
  for (let i = 0; i < m.length; i++) m[i] = sm[i] > thr ? 1 : 0;
  m = erode(dilate(m, w, h, 3), w, h, 3);
  m = fillHoles(m, w, h);
  m = keepBig(m, w, h, 0.12);
  return m;
}

export function mainTissue(m, w, h) { return keepBig(m, w, h, 0.15); }

/** (shortest, longest) caliper width in pixels — rotation-proof. */
export function feret(m, w, h) {
  const pts = [];
  for (let y = 0; y < h; y++) for (let x = 0; x < w; x++) {
    const i = y * w + x;
    if (!m[i]) continue;
    if (x === 0 || y === 0 || x === w - 1 || y === h - 1 || !m[i - 1] || !m[i + 1] || !m[i - w] || !m[i + w]) pts.push(x, y);
  }
  if (!pts.length) return [0, 0];
  let mn = Infinity, mx = 0;
  for (let a = 0; a < 180; a += 1) {
    const c = Math.cos((a * Math.PI) / 180), s = Math.sin((a * Math.PI) / 180);
    let lo = Infinity, hi = -Infinity;
    for (let i = 0; i < pts.length; i += 2) { const p = pts[i] * c + pts[i + 1] * s; if (p < lo) lo = p; if (p > hi) hi = p; }
    const wd = hi - lo + 1;
    if (wd < mn) mn = wd; if (wd > mx) mx = wd;
  }
  return [mn, mx];
}

/** Exact 2D Euclidean distance transform (Felzenszwalb): distance to nearest pixel where m=1. */
export function edt(m, w, h) {
  const INF = 1e12, f = new Float64Array(Math.max(w, h)), d = new Float64Array(Math.max(w, h));
  const v = new Int32Array(Math.max(w, h)), z = new Float64Array(Math.max(w, h) + 1);
  const g = new Float64Array(w * h);
  const dt1 = (n) => {
    let k = 0; v[0] = 0; z[0] = -INF; z[1] = INF;
    for (let q = 1; q < n; q++) {
      let s = ((f[q] + q * q) - (f[v[k]] + v[k] * v[k])) / (2 * q - 2 * v[k]);
      while (s <= z[k]) { k--; s = ((f[q] + q * q) - (f[v[k]] + v[k] * v[k])) / (2 * q - 2 * v[k]); }
      k++; v[k] = q; z[k] = s; z[k + 1] = INF;
    }
    k = 0;
    for (let q = 0; q < n; q++) { while (z[k + 1] < q) k++; d[q] = (q - v[k]) * (q - v[k]) + f[v[k]]; }
  };
  for (let x = 0; x < w; x++) {
    for (let y = 0; y < h; y++) f[y] = m[y * w + x] ? 0 : INF;
    dt1(h);
    for (let y = 0; y < h; y++) g[y * w + x] = d[y];
  }
  const out = new Float32Array(w * h);
  for (let y = 0; y < h; y++) {
    for (let x = 0; x < w; x++) f[x] = g[y * w + x];
    dt1(w);
    for (let x = 0; x < w; x++) out[y * w + x] = Math.sqrt(d[x]);
  }
  return out;
}

export function outline(m, w, h) {
  const out = new Uint8Array(w * h);
  for (let y = 1; y < h - 1; y++) for (let x = 1; x < w - 1; x++) {
    const i = y * w + x;
    if (m[i] && (!m[i - 1] || !m[i + 1] || !m[i - w] || !m[i + w])) out[i] = 1;
  }
  return out;
}
