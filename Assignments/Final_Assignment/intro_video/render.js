// Deterministic frame renderer for motion.html.
//   node render.js stills OUT_DIR t1 t2 ...      -> OUT_DIR/still_<t>.png
//   node render.js video OUT.mp4 [fps] [t0] [t1] -> H.264 (CRF 12, yuv420p) via ffmpeg
// Needs puppeteer-core (NODE_PATH) and Chrome at $CHROME (default /usr/bin/google-chrome).
const path = require("path");
const fs = require("fs");
const { spawn } = require("child_process");
const puppeteer = require("puppeteer-core");

(async () => {
  const [mode, out, ...rest] = process.argv.slice(2);
  const browser = await puppeteer.launch({
    executablePath: process.env.CHROME || "/usr/bin/google-chrome",
    headless: "new",
    args: ["--allow-file-access-from-files", "--hide-scrollbars", "--force-color-profile=srgb", "--disable-gpu"],
    defaultViewport: { width: 1920, height: 1080, deviceScaleFactor: 1 },
  });
  const page = await browser.newPage();
  page.on("pageerror", (e) => console.error("pageerror:", e.message));
  await page.goto("file://" + path.join(__dirname, "motion.html"), { waitUntil: "networkidle0" });
  await page.waitForFunction("window.READY === true", { timeout: 60000 });
  const shot = async (t) => {
    await page.evaluate((t) => window.render(t), t);
    return page.screenshot({ type: "png", clip: { x: 0, y: 0, width: 1920, height: 1080 }, optimizeForSpeed: true });
  };

  if (mode === "stills") {
    fs.mkdirSync(out, { recursive: true });
    for (const t of rest.map(Number)) fs.writeFileSync(path.join(out, `still_${t.toFixed(2)}.png`), await shot(t));
  } else {
    const fps = Number(rest[0] || 60), t0 = Number(rest[1] || 0), t1 = Number(rest[2] || 18.5);
    const n = Math.round((t1 - t0) * fps);
    const ff = spawn("ffmpeg", ["-v", "error", "-y", "-f", "image2pipe", "-framerate", String(fps), "-c:v", "png", "-i", "-",
      "-c:v", "libx264", "-preset", "slow", "-crf", "12", "-pix_fmt", "yuv420p", "-profile:v", "high",
      "-color_primaries", "bt709", "-color_trc", "bt709", "-colorspace", "bt709", "-movflags", "+faststart", out], { stdio: ["pipe", "inherit", "inherit"] });
    const started = Date.now();
    for (let i = 0; i < n; i++) {
      const buf = await shot(t0 + i / fps);
      if (!ff.stdin.write(buf)) await new Promise((r) => ff.stdin.once("drain", r));
      if (i % 120 === 0) console.log(`frame ${i}/${n}  ${((Date.now() - started) / 1000).toFixed(0)}s`);
    }
    ff.stdin.end();
    await new Promise((r) => ff.on("close", r));
    console.log(`done: ${n} frames in ${((Date.now() - started) / 1000).toFixed(0)}s -> ${out}`);
  }
  await browser.close();
})();
