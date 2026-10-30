// Loading slice images in the browser at ONE consistent working scale.
// JPG / PNG / BMP via the browser's decoder; TIFF (8/16/32-bit, multi-page, RGB) via UTIF.
import { percentiles, resample, tick } from "./imgproc.js";

const IMG_EXT = /\.(jpe?g|png|bmp|gif|webp)$/i;
const TIF_EXT = /\.tiff?$/i;

export function naturalCompare(a, b) {
  return a.localeCompare(b, undefined, { numeric: true, sensitivity: "base" });
}

export function isImageFile(f) { return IMG_EXT.test(f.name) || TIF_EXT.test(f.name); }

// ---------------------------------------------------------------- TIFF
function tiffPages(buf) {
  const ifds = UTIF.decode(buf);
  if (!ifds.length) throw new Error("No images found in this TIFF.");
  const w0 = ifds[0].t256 ? ifds[0].t256[0] : 0, h0 = ifds[0].t257 ? ifds[0].t257[0] : 0;
  // Keep full-size pages only (skip thumbnails / pyramid levels).
  return ifds.filter((d) => d.t256 && d.t256[0] === w0 && d.t257[0] === h0).slice(0, 8);
}

function tiffPageChannels(buf, ifd) {
  UTIF.decodeImage(buf, ifd);
  const w = ifd.width, h = ifd.height, n = w * h;
  const bps = ifd.t258 ? ifd.t258[0] : 8;
  const spp = ifd.t277 ? ifd.t277[0] : 1;
  const fmt = ifd.t339 ? ifd.t339[0] : 1;
  const data = ifd.data;
  let samples;
  if (bps === 8) samples = data;
  else if (bps === 16) samples = new Uint16Array(data.buffer, data.byteOffset, (data.byteLength / 2) | 0);
  else if (bps === 32 && fmt === 3) samples = new Float32Array(data.buffer.slice(data.byteOffset, data.byteOffset + data.byteLength));
  else if (bps === 32) samples = new Uint32Array(data.buffer.slice(data.byteOffset, data.byteOffset + data.byteLength));
  else if (bps === 1) { // bilevel
    const out = new Float32Array(n), bpl = Math.ceil(w / 8);
    for (let y = 0; y < h; y++) for (let x = 0; x < w; x++) out[y * w + x] = (data[y * bpl + (x >> 3)] >> (7 - (x & 7))) & 1;
    return [out];
  } else throw new Error(`Unsupported TIFF bit depth (${bps}).`);
  const used = Math.min(spp, 3);
  const planar = ifd.t284 && ifd.t284[0] === 2;   // channels stored one after another
  const chans = [];
  for (let c = 0; c < used; c++) {
    const a = new Float32Array(n);
    if (planar) for (let i = 0; i < n; i++) a[i] = samples[c * n + i];
    else for (let i = 0; i < n; i++) a[i] = samples[i * spp + c];
    chans.push(a);
  }
  ifd.data = null;
  return chans;
}

function tiffPixelSizeUm(ifd) {
  try {
    const desc = ifd.t270 ? String(ifd.t270[0] || "") : "";
    const ome = desc.match(/PhysicalSizeX="([0-9.eE+-]+)"/);
    if (ome) {
      const unit = (desc.match(/PhysicalSizeXUnit="([^"]+)"/) || [])[1] || "µm";
      return parseFloat(ome[1]) * ({ nm: 1e-3, mm: 1e3, m: 1e6 }[unit] || 1);
    }
    if (ifd.t282 && /unit=(micron|um|µm|\\u00B5m)/i.test(desc)) {
      const r = Array.isArray(ifd.t282[0]) ? ifd.t282[0][0] / ifd.t282[0][1] : ifd.t282[0];
      if (r > 0) return 1 / r;
    }
  } catch (e) { /* no size */ }
  return null;
}

// ---------------------------------------------------------------- generic
async function probe(file) {
  if (TIF_EXT.test(file.name)) {
    const buf = await file.arrayBuffer();
    const pages = tiffPages(buf);
    return { w: pages[0].t256[0], h: pages[0].t257[0], tiff: true, buf, pages };
  }
  const bmp = await createImageBitmap(file);
  const r = { w: bmp.width, h: bmp.height, tiff: false };
  bmp.close();
  return r;
}

/** Decode one file straight to working size: [channels Float32Array], nw, nh. */
async function decodeScaled(file, info, nw, nh) {
  if (info.tiff) {
    let chans = [];
    for (const page of info.pages) {
      for (const ch of tiffPageChannels(info.buf, page)) chans.push(resample(ch, info.w, info.h, nw, nh));
      await tick();
      if (chans.length >= 8) break;
    }
    return { chans, pixelUm: tiffPixelSizeUm(info.pages[0]) };
  }
  const bmp = await createImageBitmap(file);
  const cv = new OffscreenCanvas(nw, nh);
  const ctx = cv.getContext("2d", { willReadFrequently: true });
  ctx.imageSmoothingEnabled = true; ctx.imageSmoothingQuality = "high";
  // Downscale in halving steps for good area averaging.
  let src = bmp, sw = bmp.width, sh = bmp.height;
  while (sw / 2 > nw && sh / 2 > nh) {
    const tw = Math.max(nw, Math.round(sw / 2)), th = Math.max(nh, Math.round(sh / 2));
    const t = new OffscreenCanvas(tw, th), tc = t.getContext("2d");
    tc.imageSmoothingEnabled = true; tc.imageSmoothingQuality = "high";
    tc.drawImage(src, 0, 0, tw, th); src = t; sw = tw; sh = th;
  }
  ctx.drawImage(src, 0, 0, nw, nh);
  bmp.close();
  const px = ctx.getImageData(0, 0, nw, nh).data;
  const chans = [0, 1, 2].map(() => new Float32Array(nw * nh));
  for (let i = 0, j = 0; i < nw * nh; i++, j += 4) { chans[0][i] = px[j]; chans[1][i] = px[j + 1]; chans[2][i] = px[j + 2]; }
  return { chans, pixelUm: null };
}

function signal(ch) {
  const [md, hi] = percentiles(ch, [50, 99.9], 60000);
  return hi - md;
}

/** Drop channels blank in (almost) every image; find images with no signal at all. */
export function pruneChannels(slices, names) {
  const S = slices.map((s) => s.chans.map(signal));
  const nC = names.length, need = Math.max(2, Math.ceil(0.1 * slices.length));
  const keep = [];
  for (let c = 0; c < nC; c++) {
    let good = 0;
    for (const row of S) { const m = Math.max(...row, 1e-6); if (row[c] / m >= 0.25) good++; }
    if (good >= need) keep.push(c);
  }
  const k = keep.length ? keep : names.map((_, i) => i);
  const dropped = names.filter((_, i) => !k.includes(i));
  const strength = S.map((row) => Math.max(...k.map((c) => row[c])));
  const med = [...strength].sort((a, b) => a - b)[strength.length >> 1];
  const empty = strength.map((v, i) => (v < 0.2 * med ? i : -1)).filter((i) => i >= 0);
  if (dropped.length) for (const s of slices) s.chans = k.map((c) => s.chans[c]);
  return { names: k.map((c) => names[c]), dropped, empty };
}

/**
 * Load files at one common scale and pad onto one square canvas.
 * maxSize = longest side (px) of the largest image after downscaling.
 */
export async function loadStack(files, maxSize, progress = () => {}) {
  const infos = [];
  for (let i = 0; i < files.length; i++) {
    infos.push(await probe(files[i]));
    progress((10 * (i + 1)) / files.length, `Reading ${files[i].name}`);
  }
  const largest = Math.max(...infos.map((i) => Math.max(i.w, i.h)));
  const factor = largest > maxSize ? largest / maxSize : 1;
  const loaded = [];
  let pixelUm = null;
  for (let i = 0; i < files.length; i++) {
    const info = infos[i];
    const nw = Math.max(8, Math.round(info.w / factor)), nh = Math.max(8, Math.round(info.h / factor));
    const { chans, pixelUm: px } = await decodeScaled(files[i], info, nw, nh);
    if (px && !pixelUm) pixelUm = px;
    info.buf = null; info.pages = null;
    loaded.push({ name: files[i].name, w: nw, h: nh, chans });
    progress(10 + (75 * (i + 1)) / files.length, `Loaded ${files[i].name}`);
    await tick();
  }
  const nC = Math.max(...loaded.map((s) => s.chans.length));
  for (const s of loaded) while (s.chans.length < nC) s.chans.push(new Float32Array(s.w * s.h));
  const allBitmap = infos.every((i) => !i.tiff);
  let names = nC === 1 ? ["Signal"] : Array.from({ length: nC }, (_, i) => `Channel ${i + 1}`);
  if (allBitmap && nC === 3) names = ["Red", "Green", "Blue"];
  let dropped = [], empty = [];
  if (loaded.length >= 2) ({ names, dropped, empty } = pruneChannels(loaded, names));

  // Common square canvas with a margin so rotation never clips tissue.
  const maxH = Math.max(...loaded.map((s) => s.h)), maxW = Math.max(...loaded.map((s) => s.w));
  const side = Math.ceil(Math.max(maxH, maxW) * 1.3);
  const slices = loaded.map((s) => {
    const y0 = ((side - s.h) / 2) | 0, x0 = ((side - s.w) / 2) | 0;
    const chans = s.chans.map((ch) => {
      const bg = percentiles(ch, [30], 60000)[0];   // slide background level
      const out = new Float32Array(side * side).fill(bg);
      for (let y = 0; y < s.h; y++) out.set(ch.subarray(y * s.w, (y + 1) * s.w), (y + y0) * side + x0);
      return out;
    });
    return { name: s.name, chans, origW: Math.round(s.w * factor), origH: Math.round(s.h * factor) };
  });
  progress(100, "Loaded");
  return { slices, size: side, factor, names, dropped, empty, pixelUm };
}

export const DEFAULT_COLORS = [[255, 60, 60], [70, 230, 110], [90, 150, 255], [255, 80, 255], [80, 230, 255], [255, 220, 80]];
export function colorsFor(names) {
  const named = { red: [255, 60, 60], green: [70, 230, 110], blue: [90, 150, 255] };
  return names.map((n, i) => named[n.toLowerCase()] || DEFAULT_COLORS[i % DEFAULT_COLORS.length]);
}
