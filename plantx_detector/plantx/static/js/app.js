(() => {
  "use strict";
  const $ = (id) => document.getElementById(id);
  const drop = $("drop"), fileInput = $("file"), preview = $("preview"), hint = $("dropHint");
  const analyzeBtn = $("analyze"), resetBtn = $("reset"), errBox = $("error");
  const MAX_BYTES = 8 * 1024 * 1024;
  let currentFile = null;
  let previewUrl = null;

  /* ---------- model status pill ---------- */
  async function checkHealth() {
    const pill = $("modelStatus"), label = pill.querySelector("em");
    try {
      const res = await fetch("/api/health");
      const data = await res.json();
      pill.className = "status " + (data.model_loaded ? "status--ok" : "status--off");
      label.textContent = data.model_loaded ? "Model online" : "Model offline";
      pill.title = data.model_loaded ? "Model loaded and ready" : (data.error || "Model not loaded");
    } catch (e) {
      pill.className = "status status--off";
      label.textContent = "Server offline";
    }
  }

  /* ---------- helpers ---------- */
  function showError(msg) { errBox.textContent = msg; errBox.hidden = !msg; }

  function setFile(file) {
    showError("");
    if (!file) return;
    if (!/^image\//.test(file.type)) return showError("Please choose an image file.");
    if (file.size > MAX_BYTES) return showError("Image is too large (max 8 MB).");
    currentFile = file;
    if (previewUrl) URL.revokeObjectURL(previewUrl);
    previewUrl = URL.createObjectURL(file);
    preview.src = previewUrl;
    preview.hidden = false;
    hint.hidden = true;
    analyzeBtn.disabled = false;
    resetBtn.disabled = false;
  }

  function clearAll() {
    currentFile = null;
    fileInput.value = "";
    if (previewUrl) URL.revokeObjectURL(previewUrl);
    previewUrl = null;
    preview.hidden = true; preview.removeAttribute("src");
    hint.hidden = false;
    analyzeBtn.disabled = true; resetBtn.disabled = true;
    showError("");
    $("out").hidden = true; $("empty").hidden = false;
  }

  function setLoading(on) {
    analyzeBtn.disabled = on || !currentFile;
    analyzeBtn.querySelector(".spin").hidden = !on;
    analyzeBtn.querySelector(".lbl").textContent = on ? "Analysing…" : "Analyse leaf";
  }

  function fillList(ul, items) {
    ul.replaceChildren(...(items || []).map((t) => {
      const li = document.createElement("li"); li.textContent = t; return li;
    }));
  }

  /* ---------- render result ---------- */
  function render(r) {
    $("empty").hidden = true; $("out").hidden = false;
    $("verdict").classList.toggle("is-disease", r.status === "disease");
    $("pill").textContent = r.status === "healthy" ? "Healthy" : "Disease detected";
    $("rTitle").textContent = r.title;
    $("rSummary").textContent = r.summary;
    $("confVal").textContent = r.confidence.toFixed(1) + "%";
    $("ring").style.setProperty("--p", r.confidence);
    $("lowConf").hidden = !r.low_confidence;
    fillList($("rSymptoms"), r.symptoms);
    fillList($("rActions"), r.actions);

    const bars = $("bars"); bars.replaceChildren();
    r.probabilities.forEach((p, i) => {
      const row = document.createElement("div");
      row.className = "bar" + (i === 0 ? " top" : "");
      const name = document.createElement("span"); name.textContent = p.title;
      const track = document.createElement("div"); track.className = "bar__track";
      const fill = document.createElement("div"); fill.className = "bar__fill"; track.appendChild(fill);
      const val = document.createElement("span"); val.textContent = p.value.toFixed(1) + "%";
      row.append(name, track, val); bars.appendChild(row);
      requestAnimationFrame(() => requestAnimationFrame(() => { fill.style.width = p.value + "%"; }));
    });
    $("result").scrollIntoView({ behavior: "smooth", block: "nearest" });
  }

  /* ---------- call the Flask API ---------- */
  async function analyse() {
    if (!currentFile) return;
    showError(""); setLoading(true);
    const form = new FormData(); form.append("image", currentFile);
    try {
      const res = await fetch("/api/predict", { method: "POST", body: form });
      const data = await res.json().catch(() => ({}));
      if (!res.ok) throw new Error(data.error || "Something went wrong (" + res.status + ").");
      render(data);
    } catch (e) {
      showError(e.message || "Could not reach the server.");
    } finally {
      setLoading(false);
    }
  }

  /* ---------- events ---------- */
  drop.addEventListener("click", () => fileInput.click());
  drop.addEventListener("keydown", (e) => {
    if (e.key === "Enter" || e.key === " ") { e.preventDefault(); fileInput.click(); }
  });
  fileInput.addEventListener("change", () => setFile(fileInput.files[0]));
  ["dragenter", "dragover"].forEach((ev) =>
    drop.addEventListener(ev, (e) => { e.preventDefault(); drop.classList.add("over"); }));
  ["dragleave", "drop"].forEach((ev) =>
    drop.addEventListener(ev, (e) => { e.preventDefault(); drop.classList.remove("over"); }));
  drop.addEventListener("drop", (e) => setFile(e.dataTransfer.files[0]));
  analyzeBtn.addEventListener("click", analyse);
  resetBtn.addEventListener("click", clearAll);

  checkHealth();
})();
