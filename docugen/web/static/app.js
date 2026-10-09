// DocuGen AI front-end
const $ = (s) => document.querySelector(s);
const esc = (s) => String(s ?? "").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
const SWATCH = {
  cinematic: "linear-gradient(135deg,#1d2b3a,#c98b4a)", golden: "linear-gradient(135deg,#7a3b12,#f2b45a,#ffe2a1)",
  archival: "linear-gradient(135deg,#5b4a30,#d9c79e)", monochrome: "linear-gradient(135deg,#0c0c0c,#9a9a9a,#eeeeee)",
  noir: "linear-gradient(135deg,#05070a,#2b3340,#5c6a80)", painterly: "linear-gradient(135deg,#30406b,#c25f4f,#e8c46a)",
  watercolor: "linear-gradient(135deg,#9cc8e8,#f3d7e4,#f6e7b8)", animated3d: "linear-gradient(135deg,#3b6cff,#ff7ab6,#ffd36b)",
};
const ICON = {
  long: '<svg viewBox="0 0 64 40" aria-hidden="true"><rect x="2" y="4" width="60" height="32" rx="4"/><path d="M28 14l10 6-10 6z"/></svg>',
  reel: '<svg viewBox="0 0 40 64" aria-hidden="true"><rect x="6" y="2" width="28" height="60" rx="5"/><path d="M17 26l10 6-10 6z"/></svg>',
};
let cfg = null, timer = null, current_id = null, estTimer = null;
const pick = { format: "long", seconds: 60, style: "cinematic", tone: "informative", audience: "general",
  pacing: "balanced", music: "ambient", captions: "", render: "economy" };

async function api(path, opts) {
  const res = await fetch(path, opts);
  if (!res.ok) {
    let msg = res.statusText;
    try { msg = (await res.json()).detail || msg; } catch (_) {}
    if (Array.isArray(msg)) msg = msg.map((m) => m.msg).join("; ");
    throw new Error(msg);
  }
  return res.json();
}
const fileUrl = (id, p, dl) => `/api/jobs/${encodeURIComponent(id)}/files/${p}${dl ? "?download=true" : ""}`;
const money = (v) => v == null ? "-" : v < 0.01 ? "< $0.01" : "$" + v.toFixed(2);

// generic segmented control: el, entries [[key,label]], key in pick
function seg(el, entries, key, cls = "") {
  el.innerHTML = entries.map(([k, l]) => `<button type="button" data-k="${esc(k)}" class="${cls} ${String(pick[key]) === String(k) ? "on" : ""}">${l}</button>`).join("");
  el.onclick = (e) => {
    const b = e.target.closest("button"); if (!b) return;
    pick[key] = isNaN(+b.dataset.k) ? b.dataset.k : +b.dataset.k;
    [...el.children].forEach((x) => x.classList.toggle("on", x === b));
    if (key === "format") setFormat(pick.format);
    else if (key === "seconds") setScenes(fmt().default_scenes[pick.seconds]);
    else if (key === "style") paintPreview();
    estimate();
  };
}
const fmt = () => cfg.formats[pick.format];

// ------------------------------------------------------------------ home
async function loadConfig() {
  cfg = await api("/api/config");
  $("#formats").innerHTML = Object.entries(cfg.formats).map(([k, f]) =>
    `<button type="button" class="fmt ${k === pick.format ? "on" : ""}" data-k="${k}">${ICON[k] || ""}
      <span><b>${esc(f.label)}</b><small>${esc(f.aspect)} &middot; ${esc(f.hint)}</small></span></button>`).join("");
  $("#formats").onclick = (e) => {
    const b = e.target.closest("button"); if (!b) return;
    pick.format = b.dataset.k;
    [...$("#formats").children].forEach((x) => x.classList.toggle("on", x === b));
    setFormat(pick.format); estimate();
  };
  seg($("#tone"), Object.entries(cfg.tones), "tone", "pill");
  seg($("#audience"), Object.entries(cfg.audiences), "audience", "pill");
  seg($("#pacing"), Object.entries(cfg.pacing), "pacing");
  seg($("#music"), Object.entries(cfg.music), "music");
  $("#styles").innerHTML = Object.entries(cfg.styles).map(([k, l]) =>
    `<button type="button" class="style ${k === pick.style ? "on" : ""}" data-k="${k}" style="--sw:${SWATCH[k] || "#333"}"><span>${esc(l)}</span></button>`).join("");
  $("#styles").onclick = (e) => {
    const b = e.target.closest("button"); if (!b) return;
    pick.style = b.dataset.k;
    [...$("#styles").children].forEach((x) => x.classList.toggle("on", x === b));
    paintPreview();
  };
  $("#render").innerHTML = Object.entries(cfg.render).map(([k, r]) =>
    `<button type="button" data-k="${k}" class="rmode ${k === pick.render ? "on" : ""}"><b>${esc(r.label)}</b><small>${esc(r.hint)}</small></button>`).join("");
  $("#render").onclick = (e) => {
    const b = e.target.closest("button"); if (!b) return;
    pick.render = b.dataset.k;
    [...$("#render").children].forEach((x) => x.classList.toggle("on", x === b));
    estimate();
  };
  $("#voice").innerHTML = Object.entries(cfg.voices).map(([k, l]) =>
    `<option value="${esc(k)}" ${k === cfg.default_voice ? "selected" : ""}>${esc(l)}</option>`).join("");
  $("#codeRow").classList.toggle("hidden", !cfg.access_required);
  const left = Math.max(0, cfg.daily_limit - cfg.used_today);
  $("#quota").textContent = `${left} of ${cfg.daily_limit} films left today`;
  setFormat(pick.format);
}

function setFormat(k) {
  const f = cfg.formats[k];
  if (!(pick.seconds in f.lengths)) pick.seconds = f.default_seconds;
  seg($("#lengths"), Object.entries(f.lengths), "seconds");
  pick.captions = f.captions;
  seg($("#captions"), Object.entries(cfg.captions), "captions");
  $("#sceneCount").min = f.scenes.min; $("#sceneCount").max = f.scenes.max;
  setScenes(f.default_scenes[pick.seconds]);
  $("#frame").className = "frame " + k;
  $("#frameLbl").textContent = f.aspect;
  paintPreview();
}
function paintPreview() { $("#frame").style.background = SWATCH[pick.style] || "#333"; }
function setScenes(n) {
  const f = fmt();
  n = Math.max(f.scenes.min, Math.min(f.scenes.max, n || f.scenes.min));
  $("#sceneCount").value = n; $("#sceneCountOut").textContent = n;
  const maxHero = Math.min(n, f.max_heroes);
  $("#motion").max = maxHero;
  if (+$("#motion").value > maxHero) $("#motion").value = maxHero;
  $("#motionOut").textContent = $("#motion").value;
  estimate();
}
$("#sceneCount").addEventListener("input", () => setScenes(+$("#sceneCount").value));
$("#motion").addEventListener("input", () => { $("#motionOut").textContent = $("#motion").value; estimate(); });
$("#chips").addEventListener("click", (e) => { const b = e.target.closest("button"); if (b) $("#topic").value = b.textContent; });

function body() {
  return { topic: $("#topic").value.trim() || "preview topic", format: pick.format, seconds: pick.seconds,
    scenes: +$("#sceneCount").value, motion: +$("#motion").value, style: pick.style, voice: $("#voice").value,
    tone: pick.tone, audience: pick.audience, pacing: pick.pacing, captions: pick.captions, music: pick.music,
    render: pick.render, fact_check: $("#factCheck").checked, key_points: $("#keyPoints").value.trim(), code: $("#code").value };
}
function estimate() {
  clearTimeout(estTimer);
  estTimer = setTimeout(async () => {
    try {
      const e = await api("/api/estimate", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body()) });
      $("#estShots").textContent = `${e.shots}`;
      $("#estTime").textContent = `~${Math.max(2, Math.round(e.minutes))} min`;
      $("#estCost").textContent = "≈ " + money(e.usd);
      const parts = [["motion", "Motion"], ["images", "Images"], ["voice", "Voice"], ["cpu", "Editing"]];
      $("#costBar").innerHTML = parts.map(([k, l]) => {
        const v = e.breakdown[k] || 0, w = e.usd ? Math.max(2, 100 * v / e.usd) : 0;
        return `<i class="c-${k}" style="width:${w}%" title="${l}: ${money(v)}"></i>`;
      }).join("") + `<div class="legend">${parts.map(([k, l]) => `<span class="c-${k}">${l} ${money(e.breakdown[k])}</span>`).join("")}</div>`;
    } catch (_) {}
  }, 150);
}

$("#form").addEventListener("submit", async (e) => {
  e.preventDefault();
  const err = $("#formErr"); err.classList.add("hidden");
  const b = body();
  if ($("#topic").value.trim().length < 3) { err.textContent = "Please type a topic."; err.classList.remove("hidden"); $("#topic").focus(); return; }
  const go = $("#go"); go.disabled = true; go.textContent = "Starting…";
  try {
    const { id } = await api("/api/jobs", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(b) });
    location.hash = `#/job/${id}`;
  } catch (ex) { err.textContent = ex.message; err.classList.remove("hidden"); }
  finally { go.disabled = false; go.textContent = "Generate film"; }
});

async function loadRecent() {
  try {
    const items = await api("/api/recent");
    $("#recentWrap").classList.toggle("hidden", !items.length);
    $("#recent").innerHTML = items.map((j) => `<a class="rcard ${j.format === "reel" ? "reel" : ""}" href="#/job/${esc(j.id)}">
      <img loading="lazy" src="${fileUrl(j.id, "output/thumbnail.jpg")}" alt="">
      <div><b>${esc(j.title)}</b><small>${j.format === "reel" ? "Reel" : "Long video"} · ${Math.round(j.duration || 0)} s${j.cost ? " · " + money(j.cost) : ""}</small></div></a>`).join("");
  } catch (_) {}
}

// ------------------------------------------------------------------ job view
function render(job) {
  const pct = Math.round((job.progress || 0) * 100);
  const reel = (job.format || job.params?.format) === "reel";
  $("#pct").textContent = pct + "%";
  $("#bar").style.width = pct + "%";
  $("#jobTopic").textContent = `${reel ? "Reel" : "Long video"} · ${job.params?.topic || ""}`;
  $("#jobTitle").textContent = job.title || (job.status === "queued" ? "Waiting for a worker…" : "Writing the script…");
  $("#jobLogline").textContent = job.logline || "";
  $("#msg").textContent = job.message || "";
  $("#stages").innerHTML = Object.values(job.stages).map((s) => {
    const icon = s.status === "done" ? "✓" : s.status === "error" ? "!" : "";
    const right = s.status === "running" ? Math.round((s.progress || 0) * 100) + "%" : s.seconds ? Math.round(s.seconds) + " s" : s.status === "skipped" ? "skipped" : "";
    return `<li class="${esc(s.status)}"><span class="dot">${icon}</span>${esc(s.label)}<small>${right}</small></li>`;
  }).join("");
  $("#log").textContent = (job.log || []).join("\n");
  const err = $("#jobErr");
  err.classList.toggle("hidden", job.status !== "error");
  err.textContent = job.error ? `Something went wrong: ${job.error}` : "";

  // cost: estimate first, measured bill when done
  const est = job.estimate, bill = job.cost;
  $("#costBox").innerHTML = (bill ? `<p class="big">${money(bill.usd)} <small>measured GPU + CPU cost</small></p>
      <ul class="kv">${Object.entries(bill.breakdown).map(([k, v]) => `<li><span>${esc(k)}${bill.gpus?.[k] ? ` <small>${esc(bill.gpus[k].replace("NVIDIA ", ""))}</small>` : ""}</span><b>${money(v)}</b></li>`).join("")}</ul>` : "")
    + (est ? `<p class="hint">Estimate before the run: ${money(est.usd)} for ${est.shots} shots.</p>` : "");

  const fc = job.fact_check;
  $("#factPanel").classList.toggle("hidden", fc == null);
  if (fc) $("#factList").innerHTML = fc.length ? fc.map((c) => `<li>${esc(c)}</li>`).join("") : "<li>No corrections needed.</li>";
  const bible = job.director?.bible;
  $("#biblePanel").classList.toggle("hidden", !bible || !Object.keys(bible).length);
  if (bible) {
    const rows = [["Era", bible.era], ["Places", (bible.places || []).join(", ")], ["Palette", bible.palette], ["Light", bible.light], ["Motif", bible.motif]]
      .filter(([, v]) => v);
    const chars = (bible.characters || []).map((c) => `<li><b>${esc(c.name)}</b> ${esc(c.look)}</li>`).join("");
    const st = job.director.stats;
    $("#bible").innerHTML = `<ul class="kv">${rows.map(([k, v]) => `<li><span>${k}</span><b>${esc(v)}</b></li>`).join("")}</ul>`
      + (chars ? `<p class="lbl">Recurring characters</p><ul class="chars">${chars}</ul>` : "")
      + (st && st.flagged != null ? `<p class="hint">${st.shots} prompts written, ${st.flagged} flagged by the critic, ${st.repaired} repaired.</p>` : "");
  }

  const box = $("#scenes");
  box.classList.toggle("reel", reel);
  $("#scenesPanel").classList.toggle("hidden", !(job.scenes || []).length);
  (job.scenes || []).forEach((s, i) => {
    let el = box.children[i];
    if (!el) {
      el = document.createElement("div"); el.className = "scene";
      el.innerHTML = `<div class="media"><span class="tag">Scene ${i + 1}</span></div><div class="strip"></div><p></p><details class="prompts"><summary>Director's prompts</summary><ol></ol></details>`;
      box.appendChild(el);
    }
    el.querySelector("p").textContent = s.narration;
    const ol = el.querySelector(".prompts ol");
    if ((s.prompts || []).length && ol.children.length !== s.prompts.length)
      ol.innerHTML = s.prompts.map((p) => `<li>${esc(p)}</li>`).join("");
    el.querySelector(".prompts").classList.toggle("hidden", !(s.prompts || []).length);
    const media = el.querySelector(".media");
    if (s.image && !media.dataset.img) {
      media.dataset.img = 1; media.classList.add("ready");
      media.insertAdjacentHTML("afterbegin", `<img src="${fileUrl(job.id, s.image)}" alt="">`);
    }
    const strip = el.querySelector(".strip");
    (s.shots || []).slice(strip.children.length).forEach((p) =>
      strip.insertAdjacentHTML("beforeend", `<img loading="lazy" src="${fileUrl(job.id, p)}" alt="">`));
    if (s.motion && !media.dataset.mov) {
      media.dataset.mov = 1;
      media.insertAdjacentHTML("beforeend", `<video src="${fileUrl(job.id, s.motion)}" poster="${s.image ? fileUrl(job.id, s.image) : ""}" muted loop playsinline autoplay preload="auto"></video><span class="tag motion">Wan 2.2 motion</span>`);
    }
  });

  if (job.status === "done") {
    $("#result").classList.remove("hidden");
    $("#playerBox").classList.toggle("reel", reel);
    const v = $("#video");
    if (!v.src) { v.poster = fileUrl(job.id, "output/thumbnail.jpg"); v.src = fileUrl(job.id, "output/documentary.mp4"); }
    const nice = { "documentary.mp4": reel ? "Reel (9:16)" : "Film (16:9)", "documentary_audio.wav": "Final audio mix", "narration.wav": "Narration only",
      "subtitles.srt": "Captions", "thumbnail.jpg": "Thumbnail", "script.md": "Script + shot list", "script.json": "Script, prompts & visual bible (JSON)" };
    $("#downloads").innerHTML = (job.outputs || []).map((o) =>
      `<a href="${fileUrl(job.id, "output/" + o.name, true)}">${esc(nice[o.name] || o.name)}<small>${esc(o.name)} · ${(o.bytes / 1e6).toFixed(1)} MB</small></a>`).join("");
  }
}

async function poll() {
  try {
    const job = await api(`/api/jobs/${encodeURIComponent(current_id)}`);
    render(job);
    if (job.status === "done" || job.status === "error") { clearInterval(timer); timer = null; }
  } catch (ex) { $("#msg").textContent = ex.message; }
}

function route() {
  clearInterval(timer);
  const m = location.hash.match(/^#\/job\/([\w-]+)/);
  if (m) {
    current_id = m[1];
    $("#home").classList.add("hidden"); $("#job").classList.remove("hidden");
    $("#result").classList.add("hidden"); $("#scenes").innerHTML = ""; $("#video").removeAttribute("src");
    poll(); timer = setInterval(poll, 2500);
    window.scrollTo(0, 0);
  } else {
    $("#job").classList.add("hidden"); $("#home").classList.remove("hidden");
    if (!cfg) loadConfig().catch((e) => { $("#quota").textContent = "Could not reach the server: " + e.message; });
    loadRecent();
    if (!location.hash || location.hash === "#") window.scrollTo(0, 0);
  }
}
window.addEventListener("hashchange", route);
["#voice", "#factCheck"].forEach((s) => $(s).addEventListener("change", estimate));
route();
