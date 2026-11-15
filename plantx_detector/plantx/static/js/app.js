(() => {
  "use strict";
  const $ = (id) => document.getElementById(id);
  const drop = $("drop"), fileInput = $("file"), preview = $("preview"), hint = $("dropHint");
  const analyzeBtn = $("analyze"), resetBtn = $("reset"), errBox = $("error");
  const chatForm = $("chatForm"), chatInput = $("chatInput"), chatSend = $("chatSend");
  const chatMessages = $("chatMessages"), askChatTrigger = $("askChatTrigger");

  const MAX_BYTES = 8 * 1024 * 1024;
  let currentFile = null;
  let previewUrl = null;
  let latestDiagnosis = null;

  /* ---------- model status pill ---------- */
  async function checkHealth() {
    const pill = $("modelStatus"), label = pill ? pill.querySelector("em") : null;
    if (!pill || !label) return;
    try {
      const res = await fetch("/api/health");
      const data = await res.json();
      pill.className = "status " + (data.model_loaded ? "status--ok" : "status--off");
      const engine = data.backend ? data.backend.toUpperCase() : "AI";
      const acc = (data.metrics && data.metrics.accuracy) ? ` (${data.metrics.accuracy}%)` : "";
      label.textContent = data.model_loaded ? `Model online [${engine}${acc}]` : "Model offline";
      pill.title = data.model_loaded ? `Deep Learning model active: ${engine}` : (data.error || "Model not loaded");
    } catch (e) {
      pill.className = "status status--off";
      label.textContent = "Server offline";
    }
  }

  /* ---------- helpers ---------- */
  function showError(msg) {
    if (!errBox) return;
    errBox.textContent = msg;
    errBox.hidden = !msg;
  }

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
    latestDiagnosis = null;
    fileInput.value = "";
    if (previewUrl) URL.revokeObjectURL(previewUrl);
    previewUrl = null;
    preview.hidden = true;
    preview.removeAttribute("src");
    hint.hidden = false;
    analyzeBtn.disabled = true;
    resetBtn.disabled = true;
    showError("");
    $("out").hidden = true;
    $("empty").hidden = false;
  }

  function setLoading(on) {
    analyzeBtn.disabled = on || !currentFile;
    const spin = analyzeBtn.querySelector(".spin");
    const lbl = analyzeBtn.querySelector(".lbl");
    if (spin) spin.hidden = !on;
    if (lbl) lbl.textContent = on ? "Analysing…" : "Analyse leaf";
  }

  function fillList(ul, items) {
    if (!ul) return;
    ul.replaceChildren(...(items || []).map((t) => {
      const li = document.createElement("li");
      li.textContent = t;
      return li;
    }));
  }

  /* ---------- render result ---------- */
  function render(r) {
    latestDiagnosis = r;
    $("empty").hidden = true;
    $("out").hidden = false;
    $("verdict").classList.toggle("is-disease", r.status === "disease");
    $("pill").textContent = r.status === "healthy" ? "Healthy Leaf" : (r.status === "stress" ? "Crop Stress" : "Disease detected");
    $("rTitle").textContent = r.title;
    $("rSummary").textContent = r.summary;
    $("confVal").textContent = r.confidence.toFixed(1) + "%";
    $("ring").style.setProperty("--p", r.confidence);
    $("lowConf").hidden = !r.low_confidence;
    fillList($("rSymptoms"), r.symptoms);
    fillList($("rActions"), r.actions);

    const bars = $("bars");
    bars.replaceChildren();
    (r.probabilities || []).forEach((p, i) => {
      const row = document.createElement("div");
      row.className = "bar" + (i === 0 ? " top" : "");
      const name = document.createElement("span");
      name.textContent = p.title;
      const track = document.createElement("div");
      track.className = "bar__track";
      const fill = document.createElement("div");
      fill.className = "bar__fill";
      track.appendChild(fill);
      const val = document.createElement("span");
      val.textContent = p.value.toFixed(1) + "%";
      row.append(name, track, val);
      bars.appendChild(row);
      requestAnimationFrame(() => requestAnimationFrame(() => {
        fill.style.width = p.value + "%";
      }));
    });

    $("result").scrollIntoView({ behavior: "smooth", block: "nearest" });

    // Append a contextual note into chat if not already there
    appendBotMessage(`🔍 **Leaf Diagnosis Updated:** Detected **${r.title}** with ${r.confidence.toFixed(1)}% confidence. Ask me anything below about treatment, organic sprays, or irrigation for this condition!`);
  }

  /* ---------- call the Flask API ---------- */
  async function analyse() {
    if (!currentFile) return;
    showError("");
    setLoading(true);
    const form = new FormData();
    form.append("image", currentFile);
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

  /* ---------- Ask In Chat Functionality ---------- */
  function formatMarkdown(text) {
    if (!text) return "";
    let safe = text
      .replace(/&/g, "&amp;")
      .replace(/</g, "&lt;")
      .replace(/>/g, "&gt;");
    // Bold: **text**
    safe = safe.replace(/\*\*(.*?)\*\*/g, "<strong>$1</strong>");
    return safe;
  }

  function appendUserMessage(text) {
    if (!chatMessages) return;
    const msgDiv = document.createElement("div");
    msgDiv.className = "msg msg--user";
    const bubble = document.createElement("div");
    bubble.className = "msg__bubble";
    bubble.textContent = text;
    msgDiv.appendChild(bubble);
    chatMessages.appendChild(msgDiv);
    chatMessages.scrollTop = chatMessages.scrollHeight;
  }

  function appendBotMessage(htmlContent) {
    if (!chatMessages) return null;
    const msgDiv = document.createElement("div");
    msgDiv.className = "msg msg--bot";
    const bubble = document.createElement("div");
    bubble.className = "msg__bubble";
    bubble.innerHTML = formatMarkdown(htmlContent);
    msgDiv.appendChild(bubble);
    chatMessages.appendChild(msgDiv);
    chatMessages.scrollTop = chatMessages.scrollHeight;
    return bubble;
  }

  async function sendChatMessage(text) {
    if (!text || !text.trim()) return;
    appendUserMessage(text);

    if (chatInput) chatInput.value = "";
    if (chatSend) chatSend.disabled = true;

    // Loading bubble
    const loadingBubble = appendBotMessage("<em>Thinking like an agronomist…</em>");

    try {
      const payload = {
        message: text,
        context: latestDiagnosis ? {
          label: latestDiagnosis.label,
          title: latestDiagnosis.title,
          status: latestDiagnosis.status,
          confidence: latestDiagnosis.confidence
        } : {}
      };

      const res = await fetch("/api/chat", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload)
      });
      const data = await res.json().catch(() => ({}));

      if (!res.ok) {
        throw new Error(data.error || "Agronomist service unavailable.");
      }

      if (loadingBubble) {
        loadingBubble.innerHTML = formatMarkdown(data.reply || "No response received.");
      }
    } catch (err) {
      if (loadingBubble) {
        loadingBubble.textContent = "Sorry, I could not process your query right now. Please try again.";
      }
    } finally {
      if (chatSend) chatSend.disabled = false;
      if (chatMessages) chatMessages.scrollTop = chatMessages.scrollHeight;
    }
  }

  /* ---------- event bindings ---------- */
  drop.addEventListener("click", () => fileInput.click());
  drop.addEventListener("keydown", (e) => {
    if (e.key === "Enter" || e.key === " ") {
      e.preventDefault();
      fileInput.click();
    }
  });
  fileInput.addEventListener("change", () => setFile(fileInput.files[0]));
  ["dragenter", "dragover"].forEach((ev) =>
    drop.addEventListener(ev, (e) => { e.preventDefault(); drop.classList.add("over"); }));
  ["dragleave", "drop"].forEach((ev) =>
    drop.addEventListener(ev, (e) => { e.preventDefault(); drop.classList.remove("over"); }));
  drop.addEventListener("drop", (e) => setFile(e.dataTransfer.files[0]));
  analyzeBtn.addEventListener("click", analyse);
  resetBtn.addEventListener("click", clearAll);

  if (askChatTrigger) {
    askChatTrigger.addEventListener("click", () => {
      const chatSection = $("chat");
      if (chatSection) {
        chatSection.scrollIntoView({ behavior: "smooth" });
      }
      if (chatInput) {
        if (latestDiagnosis) {
          chatInput.value = `What should I do immediately to treat this ${latestDiagnosis.title}?`;
        }
        chatInput.focus();
      }
    });
  }

  if (chatForm) {
    chatForm.addEventListener("submit", (e) => {
      e.preventDefault();
      const val = chatInput ? chatInput.value : "";
      sendChatMessage(val);
    });
  }

  document.querySelectorAll(".pill-btn").forEach((pill) => {
    pill.addEventListener("click", () => {
      const query = pill.getAttribute("data-query");
      if (query) {
        sendChatMessage(query);
      }
    });
  });

  checkHealth();
})();
