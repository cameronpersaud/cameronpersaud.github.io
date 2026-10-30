// BrainStack 3D (web): the five-step workflow.
import { colorsFor, isImageFile, loadStack, naturalCompare } from "./imageio.js";
import { outline, percentiles, tick } from "./imgproc.js";
import { PLANES, anatomicalPosition } from "./orientation.js";
import { alignOne, alignStack, identity, outlineSignal, rotateAll, warp } from "./register.js";
import { buildVolume, detectDye, dyeCsv, dyeReport, guessDyeChannel, preprocess, referenceSlice, tissueExtents } from "./pipeline.js";
import { maskSurface, toSTL } from "./mesh.js";
import { Viewer } from "./viewer.js";

const $ = (id) => document.getElementById(id);
const S = {
  files: [], sel: -1, step: 0, reached: 0,
  stack: null, loadKey: null,
  cal: null, extents: null, ref: 0, barPts: [],
  dye: 0, colors: [], sensitivity: 1, proc: null, masks: null, preKey: null,
  tfs: null, aligned: false, alignImgs: null, signalImgs: null,
  vol: null, res: null, rep: null, viewer: null, busy: false,
};

// ---------------------------------------------------------------- status / progress
function status(msg) { $("status").textContent = msg; }
function prog(v) { const p = $("progress"); if (v == null) { p.hidden = true; return; } p.hidden = false; p.value = v; }
async function busy(label, fn) {
  S.busy = true; $("next").disabled = $("back").disabled = true; status(label); prog(0);
  try { return await fn((v, m) => { prog(v); if (m) status(m); }); }
  catch (e) { console.error(e); status("Something went wrong: " + e.message); throw e; }
  finally { S.busy = false; prog(null); $("next").disabled = false; $("back").disabled = S.step === 0; }
}
function ask(title, body) {
  return new Promise((resolve) => {
    const d = $("dlg"); $("dlg-title").textContent = title; $("dlg-body").textContent = body;
    const done = (v) => { d.close(); resolve(v); };
    $("dlg-yes").onclick = () => done(true); $("dlg-no").onclick = () => done(false);
    d.showModal();
  });
}
function download(blob, name) {
  const a = document.createElement("a"); a.href = URL.createObjectURL(blob); a.download = name;
  document.body.appendChild(a); a.click(); a.remove(); setTimeout(() => URL.revokeObjectURL(a.href), 2000);
}

// ---------------------------------------------------------------- orientation inputs
function orientation() {
  return {
    plane: document.querySelector("input[name=plane]:checked").value,
    topIsDorsal: $("top-is").value === "1",
    rightIs: $("right-is").value, nosePoints: $("nose").value, frontPoints: $("front").value,
  };
}
function filesReversed() { return $("order").value === "1"; }

function drawPlanePic() {
  const cv = $("plane-pic"), plane = orientation().plane;
  const dpr = window.devicePixelRatio || 1, cw = cv.clientWidth || 420, ch = cv.clientHeight || 120;
  cv.width = Math.round(cw * dpr); cv.height = Math.round(ch * dpr);
  const c = cv.getContext("2d"); c.setTransform(dpr, 0, 0, dpr, 0, 0);
  const W = cw, H = ch, cx = W / 2, cy = H / 2 + 6, k = Math.min(W / 300, H / 120);
  const brain = new Path2D();
  const ell = (x, y, rx, ry) => { brain.moveTo(x + rx, y); brain.ellipse(x, y, rx, ry, 0, 0, Math.PI * 2); };
  if (plane === "sagittal") { // seen from above, nose up
    ell(cx, cy + 2 * k, 34 * k, 34 * k); ell(cx, cy - 36 * k, 9 * k, 8 * k); ell(cx, cy + 38 * k, 23 * k, 9 * k);
  } else { // side view, nose left
    ell(cx - 4 * k, cy, 78 * k, 36 * k); ell(cx - 88 * k, cy + 5 * k, 16 * k, 11 * k); ell(cx + 70 * k, cy + 11 * k, 25 * k, 21 * k);
  }
  c.fillStyle = "rgba(185,199,214,.12)"; c.strokeStyle = "#b9c7d6"; c.lineWidth = 1.5; c.fill(brain, "nonzero"); c.stroke(brain);
  c.save(); c.clip(brain); c.strokeStyle = "#ff5a5a"; c.setLineDash([4, 3]);
  for (let i = 0; i < 11; i++) {
    c.beginPath();
    if (plane === "transverse") { const y = cy - 40 * k + i * 8 * k; c.moveTo(0, y); c.lineTo(W, y); }
    else if (plane === "sagittal") { const x = cx - 40 * k + i * 8 * k; c.moveTo(x, 0); c.lineTo(x, H); }
    else { const x = cx - 105 * k + i * 19 * k; c.moveTo(x, 0); c.lineTo(x, H); }
    c.stroke();
  }
  c.restore();
  c.fillStyle = "#8a96a2"; c.font = "11px 'IBM Plex Sans', sans-serif";
  c.fillText(plane === "sagittal" ? "seen from above (nose up)" : "seen from the side (nose left)", 4, 12);
  c.fillText(plane === "transverse" ? "slice 1 = top" : plane === "sagittal" ? "slice 1 = left side" : "slice 1 = front", 4, H - 4);
}

function syncOrientationUI() {
  const o = orientation(), p = PLANES[o.plane];
  $("plane-desc").textContent = p.desc;
  const idx = $("order").selectedIndex;
  $("order").innerHTML = p.orders.map((t, i) => `<option value="${i}">${t}</option>`).join("");
  $("order").selectedIndex = Math.max(0, idx);
  const show = (id, on) => { $(id).hidden = !on; $(id + "-l").hidden = !on; };
  show("top-is", o.plane !== "transverse");
  show("nose", o.plane === "sagittal");
  show("front", o.plane === "transverse");
  show("right-is", o.plane !== "sagittal");
  drawPlanePic(); renderFileList();
}

// ---------------------------------------------------------------- step 1: files
function sortFiles() {
  S.files.sort((a, b) => naturalCompare(a.name, b.name));
  if (filesReversed()) S.files.reverse();
}
function addFiles(list) {
  const fresh = [...list].filter(isImageFile).filter((f) => !S.files.some((g) => g.name === f.name && g.size === f.size));
  fresh.sort((a, b) => naturalCompare(a.name, b.name));
  if (filesReversed()) fresh.reverse();
  S.files.push(...fresh);
  if (S.sel < 0 && S.files.length) S.sel = 0;
  renderFileList(); previewFile();
  status(S.files.length ? `${S.files.length} images. Check the order and the cutting plane, then press Next.` : "Add your slice images to begin.");
}
function renderFileList() {
  const ul = $("files");
  if (!S.files.length) { ul.innerHTML = '<li class="empty">No images yet</li>'; return; }
  const [a, b] = PLANES[orientation().plane].ends, n = S.files.length;
  ul.innerHTML = S.files.map((f, i) => `<li data-i="${i}" class="${i === S.sel ? "sel" : ""}"><span class="num">${i + 1}</span>` +
    `<span class="name">${f.name.replace(/</g, "&lt;")}</span>` +
    (i === 0 ? `<span class="end">◀ ${a}</span>` : i === n - 1 ? `<span class="end">◀ ${b}</span>` : "") + "</li>").join("");
}
async function previewFile() {
  const f = S.files[S.sel];
  if (!f) return;
  try {
    const st = await loadStack([f], 700);
    const s = st.slices[0];
    drawChannels($("import-preview"), s.chans, st.size, colorsFor(st.names), {});
    $("import-info").textContent = `Slice ${S.sel + 1} of ${S.files.length} · ${f.name} · ${s.origW} × ${s.origH} px · ${st.names.length} channel(s)` +
      (st.pixelUm ? ` · pixel size in file ${st.pixelUm.toPrecision(4)} µm` : "");
  } catch (e) { $("import-info").textContent = `Could not read ${f.name}: ${e.message}`; }
}

// ---------------------------------------------------------------- drawing helpers
function autoRange(x) { const [lo, hi] = percentiles(x, [1, 99.7], 80000); let top = 0; for (let i = 0; i < x.length; i += 97) top = Math.max(top, x[i]); const full = top <= 1.01 ? 1 : top <= 255 ? 255 : top <= 4095 ? 4095 : 65535; return [lo, Math.max(hi, lo + 0.03 * full)]; }
function drawChannels(cv, chans, n, colors, o) {
  cv.width = n; cv.height = n;
  const ctx = cv.getContext("2d"), img = ctx.createImageData(n, n), d = img.data;
  const ranges = chans.map((c) => (o.raw ? [0, 1] : autoRange(c)));
  for (let i = 0; i < n * n; i++) {
    let r = 0, g = 0, b = 0;
    chans.forEach((c, k) => {
      if (o.visible && !o.visible[k]) return;
      const v = Math.min(1, Math.max(0, (c[i] - ranges[k][0]) / (ranges[k][1] - ranges[k][0] + 1e-9)));
      r += v * colors[k][0]; g += v * colors[k][1]; b += v * colors[k][2];
    });
    d[4 * i] = Math.min(255, r); d[4 * i + 1] = Math.min(255, g); d[4 * i + 2] = Math.min(255, b); d[4 * i + 3] = 255;
  }
  if (o.outline) { const ol = outline(o.outline, n, n); for (let i = 0; i < ol.length; i++) if (ol[i]) { d[4 * i] = 255; d[4 * i + 1] = 210; d[4 * i + 2] = 63; } }
  ctx.putImageData(img, 0, 0);
  if (o.points && o.points.length) {
    ctx.strokeStyle = "#ffd23f"; ctx.fillStyle = "#ffd23f"; ctx.lineWidth = Math.max(1, n / 300);
    if (o.points.length === 2) { ctx.beginPath(); ctx.moveTo(...o.points[0]); ctx.lineTo(...o.points[1]); ctx.stroke(); }
    for (const p of o.points) { ctx.beginPath(); ctx.arc(p[0], p[1], n / 120, 0, Math.PI * 2); ctx.stroke(); }
  }
}
function canvasPoint(cv, ev) {
  const r = cv.getBoundingClientRect();
  return [((ev.clientX - r.left) / r.width) * cv.width, ((ev.clientY - r.top) / r.height) * cv.height];
}

// ---------------------------------------------------------------- step 2: scale
function calMethod() { return document.querySelector("input[name=cal]:checked").value; }
function computeCal() {
  const st = S.stack, z = parseFloat($("z-um").value);
  if (!(z > 0)) throw new Error("Enter the slice spacing.");
  const m = calMethod();
  if (m === "micro") {
    const u = parseFloat($("um-px").value); if (!(u > 0)) throw new Error("Enter µm per pixel.");
    return { xy: u * st.factor, z, note: `microscope: ${u} µm/px × ${st.factor.toFixed(3)} downscale` };
  }
  if (m === "bar") {
    if (S.barPts.length !== 2) throw new Error("Click both ends of the scale bar in the image.");
    const d = Math.hypot(S.barPts[1][0] - S.barPts[0][0], S.barPts[1][1] - S.barPts[0][1]);
    return { xy: parseFloat($("bar-um").value) / d, z, note: `scale bar ${$("bar-um").value} µm = ${d.toFixed(1)} px` };
  }
  const real = parseFloat($("real-um").value), axis = $("dim").value;
  if (!S.extents || S.extents.axis !== axis) {
    S.extents = { axis, ext: tissueExtents(S.stack.slices.map((s, i) => S.rawMasks[i]), st.size, axis) };
    S.ref = referenceSlice(S.extents.ext);
  }
  const px = S.extents.ext[S.ref];
  return { xy: real / px, z, note: `tissue ${axis} on slice ${S.ref + 1} = ${px.toFixed(0)} px = ${real} µm (estimate)` };
}
function renderCal() {
  if (!S.stack) return;
  const m = calMethod();
  $("cal-micro").hidden = m !== "micro"; $("cal-tissue").hidden = m !== "tissue"; $("cal-bar").hidden = m !== "bar";
  let i = m === "bar" ? +$("bar-slice").value : S.ref;
  try {
    S.cal = computeCal();
    const c = S.cal;
    $("cal-result").innerHTML = `Each 3D pixel is <b>${c.xy.toPrecision(3)} × ${c.xy.toPrecision(3)} × ${c.z} µm</b>.<br>` +
      `<span class="hint">${c.note}. ${S.stack.slices.length} slices ≈ ${(S.stack.slices.length * c.z / 1000).toFixed(2)} mm deep.</span>`;
  } catch (e) { S.cal = null; $("cal-result").innerHTML = `<span style="color:var(--warn)">${e.message}</span>`; }
  if (m !== "bar") i = S.ref;
  const s = S.stack.slices[i];
  drawChannels($("cal-preview"), s.chans, S.stack.size, colorsFor(S.stack.names),
    { outline: m === "tissue" ? S.rawMasks[i] : null, points: m === "bar" ? S.barPts : null });
  $("cal-preview").classList.toggle("click", m === "bar");
  $("cal-caption").textContent = m === "bar" ? "Click one end of the scale bar, then the other." :
    m === "tissue" ? `Slice ${i + 1}, the largest tissue. Yellow = detected tissue edge.` : `Slice ${i + 1}.`;
}

// ---------------------------------------------------------------- step 3: dye & channels
function brainChannels() {
  const n = S.stack.names.length, b = [...Array(n).keys()].filter((c) => c !== S.dye);
  return b.length ? b : [...Array(n).keys()];
}
function dyeOnly() { const b = brainChannels(); return b.length === 1 && b[0] === S.dye; }
function renderChannels() {
  const st = S.stack;
  $("dye-ch").innerHTML = st.names.map((n, i) => `<option value="${i}">${n}</option>`).join("");
  $("dye-ch").value = S.dye;
  $("chan-list").innerHTML = st.names.map((n, i) => `<div class="row"><input type="color" id="col-${i}" value="${rgbHex(S.colors[i])}" aria-label="Colour of ${n}">` +
    `<span>${n}${i === S.dye ? " · dye" : " · brain"}</span></div>`).join("");
  st.names.forEach((_, i) => $("col-" + i).addEventListener("input", (e) => { S.colors[i] = hexRgb(e.target.value); renderPrep(); }));
  const notes = [];
  if (st.dropped.length) notes.push(`Ignored blank channel(s): ${st.dropped.join(", ")}.`);
  if (dyeOnly()) notes.push("Only the dye channel has signal, so the brain outline and alignment use the tissue's own glow.");
  $("chan-note").textContent = notes.join(" ");
}
const rgbHex = (c) => "#" + c.map((v) => v.toString(16).padStart(2, "0")).join("");
const hexRgb = (h) => [1, 3, 5].map((i) => parseInt(h.slice(i, i + 2), 16));
let prepTimer = 0;
function schedulePrep() { clearTimeout(prepTimer); prepTimer = setTimeout(renderPrep, 150); }
function renderPrep() {
  if (!S.stack) return;
  const i = +$("prep-slice").value, s = S.stack.slices[i], n = S.stack.size;
  $("prep-label").textContent = `Slice ${i + 1} / ${S.stack.slices.length}`;
  import("./imgproc.js").then(({ tissueMask }) => {
    const m = tissueMask(s.chans, n, n, brainChannels(), S.sensitivity);
    drawChannels($("prep-preview"), s.chans, n, S.colors, { outline: m });
  });
}

// ---------------------------------------------------------------- step 4: align
function prepareAlignImages() {
  const n = S.stack.size, b = brainChannels();
  S.signalImgs = S.proc.map((chs) => { const o = new Float32Array(n * n); for (const c of b) for (let k = 0; k < o.length; k++) o[k] += chs[c][k] / b.length; return o; });
  // No separate brain stain -> align on the tissue outline (dye spots change between slices).
  S.alignImgs = dyeOnly() ? S.signalImgs.map((s, i) => outlineSignal(s, S.masks[i], n)) : S.signalImgs;
  S.mirrorImgs = S.signalImgs.map((s, i) => outlineSignal(s, S.masks[i], n, 1.0));
  S.dispN = Math.min(n, 520);
  const { resample } = S._imgproc;
  S.disp = S.signalImgs.map((im) => { const r = resample(im, n, n, S.dispN, S.dispN); const [lo, hi] = percentiles(r, [1, 99.7]); for (let k = 0; k < r.length; k++) r[k] = Math.min(1, Math.max(0, (r[k] - lo) / (hi - lo + 1e-6))); return r; });
}
function dispT(t) { const k = S.dispN / S.stack.size; return { ...t, sx: t.sx * k, sy: t.sy * k }; }
function renderAlign() {
  if (!S.tfs) return;
  const i = +$("align-slice").value, N = S.tfs.length, dn = S.dispN;
  let j = $("align-cmp").value === "prev" ? i - 1 : i + 1;
  if (j < 0 || j >= N) j = $("align-cmp").value === "prev" ? i + 1 : i - 1;
  const cur = warp(S.disp[i], dn, dispT(S.tfs[i]));
  const nb = j >= 0 && j < N ? warp(S.disp[j], dn, dispT(S.tfs[j])) : new Float32Array(dn * dn);
  const cv = $("align-preview"); cv.width = dn; cv.height = dn;
  const ctx = cv.getContext("2d"), img = ctx.createImageData(dn, dn), d = img.data;
  for (let k = 0; k < dn * dn; k++) { d[4 * k] = nb[k] * 255; d[4 * k + 1] = cur[k] * 255; d[4 * k + 2] = nb[k] * 255; d[4 * k + 3] = 255; }
  ctx.putImageData(img, 0, 0);
  $("align-label").textContent = `Slice ${i + 1} / ${N}` + (j >= 0 && j < N ? ` vs ${j + 1}` : "");
  const t = S.tfs[i];
  renderThumbs();
  $("m-x").value = t.sx.toFixed(1); $("m-y").value = t.sy.toFixed(1); $("m-a").value = (((t.angle + 180) % 360 + 360) % 360 - 180).toFixed(1); $("m-f").checked = t.flip;
}
function renderThumbs() {
  const box = $("thumbs"), N = S.tfs.length, cur = +$("align-slice").value, tn = 110;
  if (box.children.length !== N) {
    box.innerHTML = S.tfs.map((_, i) => `<button type="button" data-i="${i}" aria-label="Slice ${i + 1}: mirror"><canvas width="${tn}" height="${tn}"></canvas><span>${i + 1}</span></button>`).join("");
    box.querySelectorAll("button").forEach((b) => b.addEventListener("click", () => {
      const i = +b.dataset.i; S.tfs[i] = { ...S.tfs[i], flip: !S.tfs[i].flip }; S.flipsEdited = true; S.vol = null;
      $("align-slice").value = i; renderAlign();
      status(`Slice ${i + 1} ${S.tfs[i].flip ? "mirrored" : "back to as-imaged"}. Press Auto-align to line the slices up again.`);
    }));
  }
  const { resample } = S._imgproc;
  box.querySelectorAll("button").forEach((b, i) => {
    b.setAttribute("aria-pressed", String(S.tfs[i].flip)); b.classList.toggle("cur", i === cur);
    const key = `${S.dispN}|${S.tfs[i].flip}`;
    if (b.dataset.key === key) return;
    b.dataset.key = key;
    const im = resample(S.disp[i], S.dispN, S.dispN, tn, tn), c = b.querySelector("canvas").getContext("2d"), d = c.createImageData(tn, tn);
    for (let y = 0; y < tn; y++) for (let x = 0; x < tn; x++) {
      const v = Math.sqrt(im[y * tn + (S.tfs[i].flip ? tn - 1 - x : x)]) * 255, k = 4 * (y * tn + x);  // gamma: show the tissue shape
      d.data[k] = v; d.data[k + 1] = v * 0.35; d.data[k + 2] = v * 0.35; d.data[k + 3] = 255;
    }
    c.putImageData(d, 0, 0);
  });
}
function alignOpts() { return { maxAngle: $("opt-upright").checked ? 15 : null, tryMirror: $("opt-mirror").checked }; }

// ---------------------------------------------------------------- step 5: 3D
function dyeSettings() {
  const mode = document.querySelector("input[name=dmode]:checked").value;
  return { channel: S.dye, mode, level: +$("dlevel").value, insideOnly: true, dyeOnly: dyeOnly() };
}
function viewExtent() { const v = S.vol; return [(v.W - 1) * v.xy, (v.H - 1) * v.xy, Math.max(1, v.Z - 1) * v.z]; }
async function runDye(rebuildBrain) {
  await busy("Finding the dye in 3D…", async (p) => {
    const v = S.vol;
    p(10, "Finding the dye in 3D…"); await tick();
    S.res = detectDye(v, dyeSettings());
    S.rep = dyeReport(v, S.res);
    if (S.res.settings.mode !== "manual") { $("dlevel").value = S.res.threshold; $("dlevel-v").textContent = S.res.threshold; }
    p(40, "Building the dye shape…"); await tick();
    const dye = S.res.found ? maskSurface(S.res.mask, v.Z, v.H, v.W, v.xy, v.z, 300) : null;
    if (rebuildBrain || !S.brainMesh) { p(70, "Building the brain surface…"); await tick(); S.brainMesh = maskSurface(v.mask, v.Z, v.H, v.W, v.xy, v.z, 200); }
    const centre = S.rep.found ? [S.rep.centreUm[0], (v.H - 1) * v.xy - S.rep.centreUm[1], S.rep.centreUm[2]] : null;
    S.viewer.setScene({ brain: S.brainMesh, dye, centre, extent: viewExtent(), orientation: orientation() });
    S.viewer.setOpacity("brain", $("brain-op").value / 100); S.viewer.setOpacity("dye", $("dye-op").value / 100);
    if (rebuildBrain) S.viewer.setView("3d");
    renderReport(); p(100);
  });
  status(S.rep.found ? "Done. Rotate the model, adjust the dye level, or download the results." : "No dye found at this level.");
}
function renderReport() {
  const r = S.rep, o = orientation(), v = S.vol;
  $("hud").textContent = `${PLANES[o.plane].label.toUpperCase()} · ${v.Z} SLICES · ${v.z} µm APART · ${v.xy.toPrecision(3)} µm/px`;
  if (!r.found) { $("report").innerHTML = `<p class="notice">No dye found at level ${S.res.threshold}. Choose “Set the level myself” and drag it lower.</p>`; return; }
  const an = anatomicalPosition(r, o), fmt = (x) => Math.round(x).toLocaleString("en-US");
  const [a, b] = PLANES[o.plane].ends;
  $("report").innerHTML =
    `<div class="row" style="align-items:baseline"><span class="big">${r.dyeMm3.toFixed(3)}</span><span class="legend">mm³ of dye · ${r.pctBrain.toFixed(2)}% of brain · ${r.deposits.length} piece${r.deposits.length === 1 ? "" : "s"}</span></div>` +
    `<table class="rep" style="margin-top:8px">` +
    `<tr><td>Behind the front</td><td class="v">${fmt(an.fromFront)} µm</td><td class="p">${an.fromFrontPct.toFixed(0)}%</td></tr>` +
    `<tr><td>Below the top</td><td class="v">${fmt(an.fromTop)} µm</td><td class="p">${an.fromTopPct.toFixed(0)}%</td></tr>` +
    `<tr><td>${an.side[0].toUpperCase() + an.side.slice(1)} of midline</td><td class="v">${fmt(an.midline)} µm</td><td class="p"></td></tr>` +
    `<tr><td>Slices with dye</td><td class="v">${r.firstSlice}–${r.lastSlice}</td><td class="p">of ${v.Z}</td></tr>` +
    `<tr><td>Most dye in</td><td class="v">slice ${r.mostSlice}</td><td class="p"></td></tr></table>` +
    `<div style="margin-top:10px"><div class="ends"><span>front</span><span>back</span></div><div class="bar"><i style="left:calc(${an.fromFrontPct}% - 1px)"></i></div>` +
    `<div class="ends" style="margin-top:8px"><span>top</span><span>bottom</span></div><div class="bar"><i style="left:calc(${an.fromTopPct}% - 1px)"></i></div></div>` +
    `<p class="hint">Slice 1 = ${a}, slice ${v.Z} = ${b}. Distances are from the edges of the tissue; the midline is the middle of its width${an.lrKnown ? "" : " (left/right not known)"}. Level ${S.res.threshold}/255.</p>`;
}

// ---------------------------------------------------------------- navigation
const STEP_ENTER = [
  () => {},
  () => { $("bar-slice").innerHTML = S.stack.slices.map((s, i) => `<option value="${i}">Slice ${i + 1} · ${s.name}</option>`).join(""); renderCal(); },
  () => { $("prep-slice").max = S.stack.slices.length - 1; if (!S.preKey) $("prep-slice").value = S.stack.slices.length >> 1; renderChannels(); renderPrep(); },
  () => { if (!S.tfs || S.tfs.length !== S.stack.slices.length) { S.tfs = S.stack.slices.map(() => identity()); S.aligned = false; }
    prepareAlignImages(); $("align-slice").max = S.tfs.length - 1;
    const o = orientation();
    // Mirroring is only visible in sagittal sections (the nose swaps ends). Coronal and
    // transverse brains look almost the same mirrored, so guessing would be a coin toss.
    if (!S.mirrorTouched) $("opt-mirror").checked = o.plane === "sagittal";
    $("straight-hint").textContent = `After aligning, rotate everything so ${o.plane === "transverse" ? `the nose points ${o.frontPoints}` : `the top of the brain points ${o.topIsDorsal ? "up" : "down"}`}, as you said in step 1.`;
    renderAlign(); },
  async () => {
    if (!S.viewer) S.viewer = new Viewer($("gl"), $("compass"));
    if (!S.vol) {
      await busy("Building the 3D model…", async (p) => { S.vol = await buildVolume(S.stack, S.proc, S.masks, S.tfs, S.cal, p); });
      S.brainMesh = null; await runDye(true);
    } else { S.viewer.resize(); }
  },
];

const STEP_LEAVE = [
  async () => {
    if (S.files.length < 2) { status("Please add at least two slice images."); return false; }
    const key = S.files.map((f) => f.name + f.size).join("|") + "@" + $("res").value;
    if (S.stack && S.loadKey === key) return true;
    const st = await busy("Loading slices…", (p) => loadStack(S.files, +$("res").value, p));
    if (st.empty.length) {
      const names = st.empty.map((i) => "• " + S.files[i].name).join("\n");
      const yes = await ask("Images without signal",
        `These images have almost no signal in the ${st.names.join(" / ")} channel(s) that the other images use (for example a DAPI-only or blank image):\n\n${names}\n\nLeave them out? If they stay, they will be treated as empty slices.`);
      if (yes) { S.files = S.files.filter((_, i) => !st.empty.includes(i)); S.sel = 0; renderFileList(); return STEP_LEAVE[0](); }
    }
    Object.assign(S, { flipsEdited: false, stack: st, loadKey: key, cal: null, extents: null, proc: null, masks: null, preKey: null, tfs: null, aligned: false, vol: null, barPts: [] });
    S.colors = colorsFor(st.names);
    S.dye = guessDyeChannel(st.names, st.slices[st.slices.length >> 1].chans);
    S.colors[S.dye] = [255, 60, 60];
    if (st.pixelUm) { $("um-px").value = st.pixelUm.toPrecision(4); document.querySelector("input[name=cal][value=micro]").checked = true; }
    const { tissueMask } = S._imgproc;
    S.rawMasks = [];
    await busy("Measuring tissue…", async (p) => {
      for (let i = 0; i < st.slices.length; i++) { S.rawMasks.push(tissueMask(st.slices[i].chans, st.size, st.size, null, 1)); p((100 * (i + 1)) / st.slices.length); await tick(); }
    });
    return true;
  },
  async () => { renderCal(); if (!S.cal) { status("Finish setting the scale first."); return false; } if (S.vol) { S.vol.xy = S.cal.xy; S.vol.z = S.cal.z; S.vol.cal = S.cal; S.brainMesh = null; S.vol.stale = true; } return true; },
  async () => {
    const key = `${S.loadKey}|${S.dye}|${S.sensitivity}`;
    if (S.proc && S.preKey === key) return true;
    const r = await busy("Cleaning up slices…", (p) => preprocess(S.stack, { maskChannels: brainChannels(), sensitivity: S.sensitivity, denoise: 1 }, p));
    S.proc = r.proc; S.masks = r.masks; S.preKey = key; S.vol = null;
    return true;
  },
  async () => {
    if (!S.aligned) {
      const yes = await ask("Align first?", "The slices haven't been aligned yet. Run automatic alignment now?");
      if (yes) await doAutoAlign();
    }
    return true;
  },
];

async function go(target) {
  if (S.busy || target === S.step) return;
  if (target > S.step) {
    for (let s = S.step; s < target; s++) {
      if (!(await STEP_LEAVE[s]())) return;
      S.step = s + 1; showStep();
      if (s + 1 < target) await STEP_ENTER[s + 1]();
    }
  } else { S.step = target; showStep(); }
  await STEP_ENTER[S.step]();
  if (S.step === 4 && S.vol && S.vol.stale) { S.vol.stale = false; await runDye(true); }
}
function showStep() {
  S.reached = Math.max(S.reached, S.step);
  document.querySelectorAll(".step").forEach((el, i) => el.classList.toggle("active", i === S.step));
  document.querySelectorAll("#steps button").forEach((b, i) => {
    if (i === S.step) b.setAttribute("aria-current", "step"); else b.removeAttribute("aria-current");
    b.classList.toggle("done", i < S.reached && i !== S.step);
    b.disabled = i > S.reached;
  });
  $("back").disabled = S.step === 0; $("next").hidden = S.step === 4;
  window.scrollTo(0, 0);
}

async function doAutoAlign() {
  const o = alignOpts();
  // Once the user has set mirrors by hand, keep them instead of guessing again.
  if (S.flipsEdited) o.tryMirror = false;
  S.tfs = await busy("Aligning slices…", (p) => alignStack(S.alignImgs, S.stack.size,
    { ...o, flips: S.tfs.map((t) => t.flip), mirrorImgs: S.mirrorImgs }, p));
  S.aligned = true; S.vol = null;
  const mir = S.tfs.map((t, i) => (t.flip ? i + 1 : 0)).filter(Boolean);
  $("align-status").textContent = `Aligned.${mir.length ? ` Mirrored slices: ${mir.join(", ")}.` : ""} Scrub through and fix any slice that looks off; the first and last slices most often need help.`;
  status("Aligned. Check the overlay, then press Next.");
  renderAlign();
}

// ---------------------------------------------------------------- wiring
async function init() {
  S._imgproc = await import("./imgproc.js");
  const drop = $("drop");
  $("file-input").addEventListener("change", (e) => { addFiles(e.target.files); e.target.value = ""; });
  drop.addEventListener("dragover", (e) => { e.preventDefault(); drop.classList.add("over"); });
  drop.addEventListener("dragleave", () => drop.classList.remove("over"));
  drop.addEventListener("drop", async (e) => {
    e.preventDefault(); drop.classList.remove("over");
    const out = [];
    const walk = async (entry) => {
      if (entry.isFile) out.push(await new Promise((r) => entry.file(r)));
      else if (entry.isDirectory) { const rd = entry.createReader(); let batch; do { batch = await new Promise((r) => rd.readEntries(r)); for (const en of batch) await walk(en); } while (batch.length); }
    };
    const items = [...(e.dataTransfer.items || [])].map((it) => it.webkitGetAsEntry && it.webkitGetAsEntry()).filter(Boolean);
    if (items.length) for (const it of items) await walk(it); else out.push(...e.dataTransfer.files);
    addFiles(out);
  });
  $("files").addEventListener("click", (e) => { const li = e.target.closest("li[data-i]"); if (!li) return; S.sel = +li.dataset.i; renderFileList(); previewFile(); });
  $("f-sort").onclick = () => { sortFiles(); renderFileList(); };
  $("f-up").onclick = () => { const i = S.sel; if (i > 0) { [S.files[i - 1], S.files[i]] = [S.files[i], S.files[i - 1]]; S.sel--; renderFileList(); } };
  $("f-down").onclick = () => { const i = S.sel; if (i >= 0 && i < S.files.length - 1) { [S.files[i + 1], S.files[i]] = [S.files[i], S.files[i + 1]]; S.sel++; renderFileList(); } };
  $("f-remove").onclick = () => { if (S.sel >= 0) { S.files.splice(S.sel, 1); S.sel = Math.min(S.sel, S.files.length - 1); renderFileList(); previewFile(); } };
  $("f-clear").onclick = () => { S.files = []; S.sel = -1; renderFileList(); };
  document.querySelectorAll("input[name=plane]").forEach((r) => r.addEventListener("change", () => { syncOrientationUI(); S.vol && (S.vol.stale = true); }));
  let lastOrder = "0";
  $("order").addEventListener("change", () => { if ($("order").value !== lastOrder) { S.files.reverse(); lastOrder = $("order").value; S.sel = S.files.length - 1 - S.sel; renderFileList(); } });
  for (const id of ["top-is", "nose", "front", "right-is"]) $(id).addEventListener("change", () => { if (S.vol) S.vol.stale = true; });

  document.querySelectorAll("input[name=cal]").forEach((r) => r.addEventListener("change", renderCal));
  for (const id of ["z-um", "um-px", "real-um", "bar-um"]) $(id).addEventListener("input", renderCal);
  $("dim").addEventListener("change", renderCal);
  $("bar-slice").addEventListener("change", () => { S.barPts = []; renderCal(); });
  $("cal-preview").addEventListener("click", (e) => {
    if (calMethod() !== "bar") return;
    if (S.barPts.length >= 2) S.barPts = [];
    S.barPts.push(canvasPoint($("cal-preview"), e)); renderCal();
  });

  $("dye-ch").addEventListener("change", () => {
    const old = S.dye; S.dye = +$("dye-ch").value;
    if (S.colors[S.dye][0] < 200) { [S.colors[old], S.colors[S.dye]] = [S.colors[S.dye], [255, 60, 60]]; }
    renderChannels(); renderPrep();
  });
  $("sens").addEventListener("input", () => { S.sensitivity = $("sens").value / 100; schedulePrep(); });
  $("prep-slice").addEventListener("input", schedulePrep);

  $("align-slice").addEventListener("input", renderAlign);
  $("align-cmp").addEventListener("change", renderAlign);
  $("auto-align").onclick = () => doAutoAlign();
  $("opt-mirror").addEventListener("change", () => { S.mirrorTouched = true; });
  $("no-align").onclick = () => { S.tfs = S.tfs.map((t) => identity(t.flip)); S.aligned = true; S.vol = null; $("align-status").textContent = "Using the images as they are."; renderAlign(); };
  const manual = () => {
    const i = +$("align-slice").value;
    S.tfs[i] = { sx: +$("m-x").value, sy: +$("m-y").value, angle: +$("m-a").value, flip: $("m-f").checked };
    S.vol = null; renderAlign();
  };
  for (const id of ["m-x", "m-y", "m-a"]) $(id).addEventListener("input", manual);
  $("m-f").addEventListener("change", () => { S.flipsEdited = true; manual(); });
  $("reset-one").onclick = () => { const i = +$("align-slice").value; S.tfs[i] = identity(); S.vol = null; renderAlign(); };
  $("realign-one").onclick = async () => {
    const i = +$("align-slice").value, N = S.tfs.length;
    let j = $("align-cmp").value === "prev" ? i - 1 : i + 1; if (j < 0 || j >= N) j = i === 0 ? 1 : i - 1;
    const n = S.stack.size;
    await busy(`Re-aligning slice ${i + 1}…`, async () => {
      await tick();
      const fixed = warp(S.alignImgs[j], n, S.tfs[j]);
      S.tfs[i] = alignOne(fixed, S.alignImgs[i], n, { maxAngle: alignOpts().maxAngle, tryMirror: false, flip: $("m-f").checked });
    });
    S.vol = null; renderAlign(); status(`Slice ${i + 1} re-aligned.`);
  };
  document.querySelectorAll("[data-rot]").forEach((b) => b.addEventListener("click", () => { S.tfs = rotateAll(S.tfs, +b.dataset.rot); S.vol = null; renderAlign(); }));

  document.querySelectorAll("[data-cam]").forEach((b) => b.addEventListener("click", () => S.viewer && S.viewer.setView(b.dataset.cam)));
  $("spin").onclick = () => { S.viewer.spin = !S.viewer.spin; $("spin").setAttribute("aria-pressed", String(S.viewer.spin)); };
  $("brain-op").addEventListener("input", () => S.viewer && S.viewer.setOpacity("brain", $("brain-op").value / 100));
  $("dye-op").addEventListener("input", () => S.viewer && S.viewer.setOpacity("dye", $("dye-op").value / 100));
  let dyeTimer = 0;
  const redye = () => { clearTimeout(dyeTimer); dyeTimer = setTimeout(() => runDye(false), 350); };
  document.querySelectorAll("input[name=dmode]").forEach((r) => r.addEventListener("change", redye));
  $("dlevel").addEventListener("input", () => { $("dlevel-v").textContent = $("dlevel").value; document.querySelector("input[name=dmode][value=manual]").checked = true; redye(); });
  $("dl-csv").onclick = () => download(new Blob([dyeCsv(S.vol, S.res, S.rep, orientation(), { calibration: S.cal.note, files: S.stack.slices.map((s) => s.name) })], { type: "text/csv" }), "dye_location_report.csv");
  $("dl-png").onclick = () => fetch(S.viewer.snapshot()).then((r) => r.blob()).then((b) => download(b, "dye_in_brain_3d.png"));
  $("dl-dye").onclick = () => { const m = S.viewer.worldMesh("dye"); if (m) download(toSTL(m.positions, m.indices, "dye"), "dye_shape.stl"); else status("No dye shape to download."); };
  $("dl-brain").onclick = () => { const m = S.viewer.worldMesh("brain"); if (m) download(toSTL(m.positions, m.indices, "brain"), "brain_surface.stl"); };

  document.querySelectorAll("#steps button").forEach((b) => b.addEventListener("click", () => go(+b.dataset.step)));
  $("next").onclick = () => go(S.step + 1);
  $("back").onclick = () => go(S.step - 1);
  window.addEventListener("resize", drawPlanePic);
  syncOrientationUI(); showStep();
  window.BrainStack = { S, go, doAutoAlign, addFiles }; // handy for testing and debugging
}
init();
