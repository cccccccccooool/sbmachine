import path from 'node:path';
import fs from 'node:fs';
import os from 'node:os';
import { pathToFileURL } from 'node:url';
import { execPath } from 'node:process';
import { spawn } from 'node:child_process';

const SKILL = 'C:/Users/sihasiha/.agents/skills/archify';
const vc = await import(pathToFileURL(path.join(SKILL, 'bin', 'visual-check.mjs')).href);

const ARTIFACT = path.resolve('ai6657-architecture.html');
const VIEW_IDS = ['main-path', 'llm-layer', 'optional-backends'];

const browser = new vc.ChromeVisualBrowser(vc.findChrome());
const sessionId = await browser.sessionPromise;

async function open({ width, height, dsf, theme, hash, frame = 0 }) {
  await browser.cdp.send('Emulation.setDeviceMetricsOverride', {
    width, height, deviceScaleFactor: dsf, mobile: false,
  }, sessionId);
  const url = new URL(pathToFileURL(ARTIFACT).href);
  url.searchParams.set('theme', theme);
  url.searchParams.set('embed', '1');
  if (frame > 0) url.searchParams.set('frame', String(frame));
  if (hash) url.hash = hash;
  const loaded = browser.cdp.waitFor('Page.loadEventFired', sessionId);
  const nav = await browser.cdp.send('Page.navigate', { url: url.href }, sessionId);
  if (nav.errorText) throw new Error(nav.errorText);
  await loaded;
  await new Promise((r) => setTimeout(r, 700));
  await browser.cdp.send('Runtime.evaluate', {
    expression: `(function(){
      var fontsReady = document.fonts && document.fonts.ready ? document.fonts.ready.catch(function(){}) : Promise.resolve();
      function raf2(){ return new Promise(function(res){ requestAnimationFrame(function(){ requestAnimationFrame(res); }); }); }
      var s = (window.Archify && Archify.readerLayout && Archify.readerLayout.whenStable) ? Archify.readerLayout.whenStable() : Promise.resolve();
      return s.then(raf2).then(raf2);
    })()`,
    awaitPromise: true,
    returnByValue: true,
  }, sessionId);
  await new Promise((r) => setTimeout(r, 500));
}

async function diagramClip() {
  const res = await browser.cdp.send('Runtime.evaluate', {
    expression: `(function(){
      var el = document.querySelector('.diagram-container');
      if (!el) return null;
      var r = el.getBoundingClientRect();
      return { x: Math.max(0, r.left + window.scrollX), y: Math.max(0, r.top + window.scrollY),
               width: Math.ceil(r.width), height: Math.ceil(r.height) };
    })()`,
    returnByValue: true,
  }, sessionId);
  if (!res.result?.value) throw new Error('.diagram-container not found');
  return res.result.value;
}

async function shot(clip, scale, out) {
  const cap = await browser.cdp.send('Page.captureScreenshot', {
    format: 'png', fromSurface: true, captureBeyondViewport: false,
    clip: { ...clip, scale },
  }, sessionId, 30000);
  fs.writeFileSync(out, Buffer.from(cap.data, 'base64'));
  console.log('saved', out, clip.width * scale + 'x' + clip.height * scale);
}

// 1) 高清静态图（明/暗），2x 分辨率
for (const theme of ['light', 'dark']) {
  await open({ width: 2048, height: 1320, dsf: 1, theme, hash: '' });
  const clip = await diagramClip();
  await shot(clip, 2, `ai6657-architecture.${theme}.png`);
}

// 2) GIF 帧：全图 → 三个引导视图
const frames = ['', ...VIEW_IDS.map((id) => `view=${id}`)];
for (let i = 0; i < frames.length; i++) {
  await open({ width: 1600, height: 1000, dsf: 1, theme: 'light', hash: frames[i], frame: i });
  const clip = await diagramClip();
  await shot(clip, 1, `_gif_frame_${i}.png`);
}

await browser.close();
console.log('done');
