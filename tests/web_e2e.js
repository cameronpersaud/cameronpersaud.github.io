// End-to-end test of the browser app with Playwright.
// node tests/web_e2e.js <image_folder> <out_folder> [plane] [zUm] [realUm]
const { chromium } = require("playwright");
const fs = require("fs");
const path = require("path");
const http = require("http");

const [, , imgDir, outDir, plane = "coronal", zUm = "100", realUm = "6000", upright = "0", mirrored = ""] = process.argv;
// mirrored: comma-separated slice numbers that truly face the other way (to fix by clicking thumbnails)
const DOCS = path.join(__dirname, "..", "docs");
fs.mkdirSync(outDir, { recursive: true });

const types = { ".html": "text/html", ".js": "text/javascript", ".css": "text/css", ".json": "application/json" };
const server = http.createServer((req, res) => {
  const p = path.join(DOCS, decodeURIComponent(req.url.split("?")[0]).replace(/^\/$/, "/index.html"));
  if (!p.startsWith(DOCS) || !fs.existsSync(p)) { res.writeHead(404); return res.end(); }
  res.writeHead(200, { "Content-Type": types[path.extname(p)] || "application/octet-stream" });
  fs.createReadStream(p).pipe(res);
});

(async () => {
  await new Promise((r) => server.listen(8777, r));
  const browser = await chromium.launch({ args: ["--use-gl=swiftshader", "--enable-webgl", "--ignore-gpu-blocklist"] });
  const page = await browser.newPage({ viewport: { width: 1440, height: 900 }, acceptDownloads: true });
  const errors = [];
  page.on("pageerror", (e) => errors.push(e.message));
  page.on("console", (m) => { if (m.type() === "error" && !/fonts\.g/.test(m.text())) errors.push(m.text()); });
  await page.route(/fonts\.googleapis|fonts\.gstatic/, (r) => r.fulfill({ body: "", contentType: "text/css" }));
  const t0 = Date.now();
  const lap = (msg) => console.log(`[${((Date.now() - t0) / 1000).toFixed(1)}s] ${msg}`);
  await page.goto("http://localhost:8777/");
  const files = fs.readdirSync(imgDir).filter((f) => /\.(tif|tiff|png|jpe?g)$/i.test(f)).sort().map((f) => path.join(imgDir, f));
  await page.check(`input[name=plane][value=${plane}]`);
  await page.setInputFiles("#file-input", files);
  await page.waitForTimeout(1500);
  await page.screenshot({ path: path.join(outDir, "w1_import.png") });
  const waitIdle = async () => { await page.waitForFunction(() => !window.BrainStack.S.busy, null, { timeout: 600000 }); await page.waitForTimeout(300); };

  // auto-accept the "leave out empty images" dialog
  page.on("dialog", (d) => d.accept());
  const clickYesIfOpen = async () => { if (await page.isVisible("#dlg[open]")) await page.click("#dlg-yes"); };

  await page.click("#next");
  for (let i = 0; i < 1200; i++) { await page.waitForTimeout(500); await clickYesIfOpen(); if (await page.evaluate(() => window.BrainStack.S.step === 1 && !window.BrainStack.S.busy)) break; }
  lap("loaded: " + JSON.stringify(await page.evaluate(() => { const s = window.BrainStack.S.stack; return { n: s.slices.length, size: s.size, names: s.names, dropped: s.dropped, factor: +s.factor.toFixed(3) }; })));
  await page.fill("#z-um", zUm);
  await page.fill("#real-um", realUm);
  await page.waitForTimeout(800);
  await page.screenshot({ path: path.join(outDir, "w2_scale.png") });
  lap("cal: " + JSON.stringify(await page.evaluate(() => window.BrainStack.S.cal)));

  await page.click("#next"); await waitIdle();
  lap("dye channel guess: " + (await page.evaluate(() => window.BrainStack.S.dye)));
  await page.waitForTimeout(800);
  await page.screenshot({ path: path.join(outDir, "w3_dye.png") });

  await page.click("#next"); await waitIdle();
  if (upright !== "1") await page.uncheck("#opt-upright");
  await page.click("#auto-align"); await waitIdle();
  lap("aligned: angles " + JSON.stringify(await page.evaluate(() => window.BrainStack.S.tfs.map((t) => Math.round(t.angle)))) +
      " mirrored " + JSON.stringify(await page.evaluate(() => window.BrainStack.S.tfs.map((t, i) => (t.flip ? i + 1 : 0)).filter(Boolean))));
  if (mirrored) {
    const want = mirrored.split(",").map(Number);
    const got = await page.evaluate(() => window.BrainStack.S.tfs.map((t, i) => (t.flip ? i + 1 : 0)).filter(Boolean));
    const ref = await page.evaluate(() => window.BrainStack.S.tfs.length >> 1);
    // "true" mirrors are relative to the reference slice (which is never mirrored)
    const refFlipped = want.includes(ref + 1);
    const N = await page.evaluate(() => window.BrainStack.S.tfs.length);
    const target = []; for (let i = 1; i <= N; i++) if (want.includes(i) !== refFlipped) target.push(i);
    const wrong = [...new Set([...got.filter((g) => !target.includes(g)), ...target.filter((t) => !got.includes(t))])];
    lap(`auto mirror calls: ${N - wrong.length}/${N} right; fixing slices ${wrong.join(",") || "none"} by clicking thumbnails`);
    for (const i of wrong) await page.click(`#thumbs button[data-i="${i - 1}"]`);
    await page.screenshot({ path: path.join(outDir, "w4_thumbs.png") });
    await page.click("#auto-align"); await waitIdle();
    lap("re-aligned: mirrored " + JSON.stringify(await page.evaluate(() => window.BrainStack.S.tfs.map((t, i) => (t.flip ? i + 1 : 0)).filter(Boolean))) +
        " angles " + JSON.stringify(await page.evaluate(() => window.BrainStack.S.tfs.map((t) => Math.round(t.angle)))));
  }
  await page.evaluate(() => { document.getElementById("align-slice").value = 10; document.getElementById("align-slice").dispatchEvent(new Event("input")); });
  await page.screenshot({ path: path.join(outDir, "w4_align.png") });

  await page.click("#next");
  await page.waitForFunction(() => window.BrainStack.S.rep, null, { timeout: 600000 }); await waitIdle();
  await page.waitForTimeout(1500);
  lap("report: " + JSON.stringify(await page.evaluate(() => { const r = window.BrainStack.S.rep, v = window.BrainStack.S.vol; return {
    dyeMm3: +r.dyeMm3?.toFixed(3), pct: +r.pctBrain?.toFixed(2), pieces: r.deposits.length, slices: [r.firstSlice, r.lastSlice], most: r.mostSlice,
    centreUm: r.centreUm?.map(Math.round), thr: r.threshold, vol: [v.Z, v.H, v.W], cropOffset: v.cropOffset, xy: +v.xy.toFixed(3) }; })));
  await page.screenshot({ path: path.join(outDir, "w5_view.png") });
  await page.click("[data-cam=side]"); await page.waitForTimeout(800);
  await page.screenshot({ path: path.join(outDir, "w5_side.png") });
  const [dl] = await Promise.all([page.waitForEvent("download"), page.click("#dl-csv")]);
  await dl.saveAs(path.join(outDir, "web_dye_report.csv"));
  const [dl2] = await Promise.all([page.waitForEvent("download"), page.click("#dl-dye")]);
  await dl2.saveAs(path.join(outDir, "web_dye.stl"));
  lap("downloads ok");
  console.log("errors:", errors);
  await browser.close(); server.close();
})().catch((e) => { console.error(e); process.exit(1); });
