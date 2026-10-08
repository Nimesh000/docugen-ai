// DocuGen AI front-end
const $ = (s) => document.querySelector(s);
const esc = (s) => String(s ?? "").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
const SWATCH = {
  cinematic: "linear-gradient(120deg,#2b3a4a,#c98b4a)", archival: "linear-gradient(120deg,#6b5a3e,#d9c79e)",
  painterly: "linear-gradient(120deg,#30406b,#c25f4f,#e8c46a)", noir: "linear-gradient(120deg,#05070a,#3a4250)",
};
let cfg = null, timer = null, current = null;
const pick = { seconds: 60, style: "cinematic" };

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

// ------------------------------------------------------------------ home
async function loadConfig() {
  cfg = await api("/api/config");
  $("#lengths").innerHTML = Object.entries(cfg.lengths).map(([s, l]) =>
    `<button type="button" data-s="${s}" class="${+s === pick.seconds ? "on" : ""}">${esc(l)}</button>`).join("");
  $("#styles").innerHTML = Object.entries(cfg.styles).map(([k, l]) =>
    `<button type="button" class="style ${k === pick.style ? "on" : ""}" data-k="${k}" style="--sw:${SWATCH[k] || "#333"}">${esc(l)}</button>`).join("");
  $("#voice").innerHTML = Object.entries(cfg.voices).map(([k, l]) =>
    `<option value="${esc(k)}" ${k === cfg.default_voice ? "selected" : ""}>${esc(l)}</option>`).join("");
  $("#sceneCount").min = cfg.scenes.min; $("#sceneCount").max = cfg.scenes.max;
  setScenes(cfg.scenes.default[pick.seconds] || 6);
  $("#codeRow").classList.toggle("hidden", !cfg.access_required);
  const left = Math.max(0, cfg.daily_limit - cfg.used_today);
  $("#quota").textContent = `${left} of ${cfg.daily_limit} documentaries left today`;
}
$("#lengths").addEventListener("click", (e) => {
  const b = e.target.closest("button"); if (!b) return;
  pick.seconds = +b.dataset.s;
  [...$("#lengths").children].forEach((x) => x.classList.toggle("on", x === b));
  if (cfg) setScenes(cfg.scenes.default[pick.seconds] || +$("#sceneCount").value);
});
function setScenes(n) {
  $("#sceneCount").value = n; $("#sceneCountOut").textContent = n;
  const maxHero = Math.min(n, cfg ? cfg.max_heroes : 6);
  $("#motion").max = maxHero;
  if (+$("#motion").value > maxHero) $("#motion").value = maxHero;
  $("#motionOut").textContent = $("#motion").value;
  estimate();
}
function estimate() {
  const shots = Math.max(+$("#sceneCount").value, Math.round(pick.seconds / 3.2));
  const heroes = +$("#motion").value;
  const mins = Math.max(2, Math.round(1.2 + pick.seconds / 60 + heroes * 0.25));
  $("#estimate").innerHTML = `About <b>${shots} shots</b>, ${heroes} animated &middot; ready in roughly <b>${mins}–${mins + 2} minutes</b> (first film of the day takes longer while the GPUs load the models)`;
}
$("#sceneCount").addEventListener("input", () => setScenes(+$("#sceneCount").value));
$("#styles").addEventListener("click", (e) => {
  const b = e.target.closest("button"); if (!b) return;
  pick.style = b.dataset.k;
  [...$("#styles").children].forEach((x) => x.classList.toggle("on", x === b));
});
$("#chips").addEventListener("click", (e) => { const b = e.target.closest("button"); if (b) $("#topic").value = b.textContent; });
$("#motion").addEventListener("input", () => { $("#motionOut").textContent = $("#motion").value; estimate(); });

$("#form").addEventListener("submit", async (e) => {
  e.preventDefault();
  const err = $("#formErr"); err.classList.add("hidden");
  const body = {
    topic: $("#topic").value.trim(), seconds: pick.seconds, scenes: +$("#sceneCount").value, style: pick.style, voice: $("#voice").value,
    motion: +$("#motion").value, music: $("#music").checked, subtitles: $("#subtitles").checked, code: $("#code").value,
  };
  if (body.topic.length < 3) { err.textContent = "Please type a topic."; err.classList.remove("hidden"); return; }
  const go = $("#go"); go.disabled = true; go.textContent = "Starting…";
  try {
    const { id } = await api("/api/jobs", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) });
    location.hash = `#/job/${id}`;
  } catch (ex) { err.textContent = ex.message; err.classList.remove("hidden"); }
  finally { go.disabled = false; go.textContent = "Generate documentary"; }
});

async function loadRecent() {
  try {
    const items = await api("/api/recent");
    $("#recentWrap").classList.toggle("hidden", !items.length);
    $("#recent").innerHTML = items.map((j) => `<a class="card" href="#/job/${esc(j.id)}">
      <img loading="lazy" src="${fileUrl(j.id, "output/thumbnail.jpg")}" alt="">
      <div><b>${esc(j.title)}</b><small>${Math.round(j.duration || 0)} s · ${esc(j.logline || "")}</small></div></a>`).join("");
  } catch (_) {}
}

// ------------------------------------------------------------------ job view
function render(job) {
  current = job;
  const pct = Math.round((job.progress || 0) * 100);
  $("#pct").textContent = pct + "%";
  $("#bar").style.width = pct + "%";
  $("#jobTopic").textContent = job.params?.topic || "";
  $("#jobTitle").textContent = job.title || (job.status === "queued" ? "Waiting for a GPU worker…" : "Writing the script…");
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

  const box = $("#scenes");
  $("#scenesPanel").classList.toggle("hidden", !(job.scenes || []).length);
  (job.scenes || []).forEach((s, i) => {
    let el = box.children[i];
    if (!el) {
      el = document.createElement("div"); el.className = "scene";
      el.innerHTML = `<div class="media"><span class="tag">Scene ${i + 1}</span></div><div class="strip"></div><p></p>`;
      box.appendChild(el);
    }
    el.querySelector("p").textContent = s.narration;
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
    const v = $("#video");
    if (!v.src) { v.poster = fileUrl(job.id, "output/thumbnail.jpg"); v.src = fileUrl(job.id, "output/documentary.mp4"); }
    const nice = { "documentary.mp4": "Final film", "documentary_audio.wav": "Final audio mix", "narration.wav": "Narration only",
      "subtitles.srt": "Captions", "thumbnail.jpg": "Thumbnail", "script.md": "Script", "script.json": "Script (JSON)" };
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

let current_id = null;
function route() {
  clearInterval(timer);
  const m = location.hash.match(/^#\/job\/([\w-]+)/);
  if (m) {
    current_id = m[1];
    $("#home").classList.add("hidden"); $("#job").classList.remove("hidden");
    $("#result").classList.add("hidden"); $("#scenes").innerHTML = ""; $("#video").removeAttribute("src");
    poll(); timer = setInterval(poll, 2500);
  } else {
    $("#job").classList.add("hidden"); $("#home").classList.remove("hidden");
    loadConfig().catch((e) => { $("#quota").textContent = "Could not reach the server: " + e.message; });
    loadRecent();
  }
  window.scrollTo(0, 0);
}
window.addEventListener("hashchange", route);
route();
