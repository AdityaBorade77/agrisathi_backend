/* ============================================================
   app.js — Marathi-English Speech Translator PWA
   ============================================================

   Modules:
   - Config        : Settings management (API URL, voice, HF token)
   - API           : Backend calls (speech-to-text, text-to-speech)
   - SpeechInput   : Mic recording + Web Speech API live transcript
   - TTSPlayer     : Audio playback + word highlighting
   - ChatUI        : Message rendering
   - App           : Main coordinator
   ============================================================ */

"use strict";

/* ─── Config ─────────────────────────────────────────────── */
const Config = (() => {
  // The backend address comes from config.js (edit it there when hosting).
  const CONFIG_API_URL =
    (window.AGRISAATHI_CONFIG && window.AGRISAATHI_CONFIG.apiUrl) || "http://localhost:8000";

  const DEFAULTS = {
    apiUrl: CONFIG_API_URL.replace(/\/+$/, ""),
    voice: "Sunita",
    language: "mr",
  };

  let _settings = { ...DEFAULTS };

  function load() {
    try {
      const saved = localStorage.getItem("marathi_translator_settings");
      if (saved) {
        const parsed = JSON.parse(saved);
        // Older versions saved the built-in localhost address on every save
        // (and a placeholder HF token). Drop both so config.js stays in charge.
        if (parsed.apiUrl === "http://localhost:8000") delete parsed.apiUrl;
        delete parsed.hfToken;
        _settings = { ...DEFAULTS, ...parsed };
      }
    } catch (e) {
      console.warn("Failed to load settings:", e);
    }
  }

  function save(updates) {
    _settings = { ...DEFAULTS, ..._settings, ...updates };
    // Only remember the backend address if it was changed in Settings;
    // otherwise a new address in config.js would never reach this phone.
    const toStore = { ..._settings };
    if (toStore.apiUrl === DEFAULTS.apiUrl) delete toStore.apiUrl;
    try {
      localStorage.setItem("marathi_translator_settings", JSON.stringify(toStore));
    } catch (e) {
      console.warn("Failed to save settings:", e);
    }
  }

  function get(key) { return _settings[key]; }

  // The address from config.js (used when the Settings box is left empty).
  function defaultApiUrl() { return DEFAULTS.apiUrl; }

  load();
  return { get, save, load, defaultApiUrl };
})();

/* ─── Language Selector (mr / hi / en) ───────────────────── */
const LanguageSelector = (() => {
  const _listeners = [];

  function getLanguage() {
    return Config.get("language") || "mr";
  }

  // Lets other modules (e.g. CropAdvisor) re-localize their own UI text
  // when the farmer switches language, without LanguageSelector needing
  // to know anything about them.
  function onChange(callback) {
    _listeners.push(callback);
  }

  function _applyActiveState(lang) {
    document.querySelectorAll(".lang-btn").forEach((btn) => {
      btn.classList.toggle("active", btn.dataset.lang === lang);
    });
    document.querySelectorAll(".auth-lang-btn").forEach((btn) => {
      btn.classList.toggle("active", btn.dataset.lang === lang);
    });
  }

  function setLanguage(lang) {
    Config.save({ language: lang });
    _applyActiveState(lang);
    _listeners.forEach((cb) => cb(lang));
  }

  function init() {
    document.querySelectorAll(".lang-btn").forEach((btn) => {
      btn.addEventListener("click", () => {
        setLanguage(btn.dataset.lang);
      });
    });
    _applyActiveState(getLanguage());
  }

  return { init, getLanguage, setLanguage, onChange };
})();


/* ─── API Client ─────────────────────────────────────────── */
const API = (() => {
  async function speechToText(audioBlob) {
    const apiUrl = Config.get("apiUrl");
    const formData = new FormData();
    formData.append("audio", audioBlob, "recording.wav");
    // The LLM understands Marathi directly — skip the MR→EN translation hop.
    formData.append("translate", "false");
    formData.append("language", LanguageSelector.getLanguage());

    const response = await fetch(`${apiUrl}/api/speech-to-text`, {
      method: "POST",
      body: formData,
    });

    if (!response.ok) {
      const err = await response.json().catch(() => ({}));
      throw new Error(err.detail || `API Error: ${response.status}`);
    }

    return response.json();
  }

  async function textToSpeech(text, voice, language = LanguageSelector.getLanguage()) {
    const apiUrl = Config.get("apiUrl");
    const response = await fetch(`${apiUrl}/api/text-to-speech`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ text: text, language: language, voice: voice }),
    });

    if (!response.ok) {
      const err = await response.json().catch(() => ({}));
      throw new Error(err.detail || `API Error: ${response.status}`);
    }

    return response.json();
  }

  async function chat(text) {
    const apiUrl = Config.get("apiUrl");
    const response = await fetch(`${apiUrl}/api/chat`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ text: text, reply_language: LanguageSelector.getLanguage() }),
    });

    if (!response.ok) {
      const err = await response.json().catch(() => ({}));
      throw new Error(err.detail || `API Error: ${response.status}`);
    }

    return response.json();
  }

  async function healthCheck() {
    const apiUrl = Config.get("apiUrl");
    const response = await fetch(`${apiUrl}/api/health`, {
      signal: AbortSignal.timeout(5000),
    });
    return response.json();
  }

  // Talukas, soil types and crops for the Crop Advisor form's dropdowns.
  async function cropOptions() {
    const apiUrl = Config.get("apiUrl");
    const response = await fetch(
      `${apiUrl}/api/crop/options?language=${LanguageSelector.getLanguage()}`
    );
    if (!response.ok) {
      const err = await response.json().catch(() => ({}));
      throw new Error(err.detail || `API Error: ${response.status}`);
    }
    return response.json();
  }

  // Crop recommendation + Nemotron explanation. 20s client timeout — the
  // backend's own LLM budget is 10s, so this leaves slack for crop_api and
  // network time before giving up with a friendly message.
  async function cropAdvice(payload) {
    const apiUrl = Config.get("apiUrl");
    const response = await fetch(`${apiUrl}/api/crop`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
      signal: AbortSignal.timeout(20000),
    });

    const data = await response.json().catch(() => ({}));
    if (!response.ok) {
      const errors = Array.isArray(data.errors) ? data.errors.join(" ") : null;
      const err = new Error(data.detail || errors || `API Error: ${response.status}`);
      err.status = response.status;
      throw err;
    }
    return data;
  }

  async function createChat(title) {
    const apiUrl = Config.get("apiUrl");
    const response = await fetch(`${apiUrl}/api/chats`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ title }),
    });
    return response.json();
  }

  async function getChats() {
    const apiUrl = Config.get("apiUrl");
    const response = await fetch(`${apiUrl}/api/chats`);
    return response.json();
  }

  async function getChatMessages(chatId) {
    const apiUrl = Config.get("apiUrl");
    const response = await fetch(`${apiUrl}/api/chats/${chatId}/messages`);
    return response.json();
  }

  async function saveMessage(chatId, role, msgType, content) {
    const apiUrl = Config.get("apiUrl");
    const response = await fetch(`${apiUrl}/api/chats/${chatId}/messages`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ role, msg_type: msgType, content }),
    });
    return response.json();
  }

  return {
    speechToText, textToSpeech, healthCheck, chat, createChat, getChats, getChatMessages, saveMessage,
    cropOptions, cropAdvice,
  };
})();

/* ─── Disease Check (leaf photo upload) ──────────────────── */
const DiseaseCheck = (() => {
  const MAX_SIDE = 1280;
  const JPEG_QUALITY = 0.85;

  // Shrink the photo in the browser before upload: at most 1280px on the
  // longer side, re-encoded as JPEG, so slow mobile uploads stay fast.
  function _compressImage(file) {
    return new Promise((resolve, reject) => {
      const img = new Image();
      const url = URL.createObjectURL(file);

      img.onload = () => {
        URL.revokeObjectURL(url);
        let { width, height } = img;
        const longSide = Math.max(width, height);
        if (longSide > MAX_SIDE) {
          const scale = MAX_SIDE / longSide;
          width = Math.round(width * scale);
          height = Math.round(height * scale);
        }

        const canvas = document.createElement("canvas");
        canvas.width = width;
        canvas.height = height;
        canvas.getContext("2d").drawImage(img, 0, 0, width, height);
        canvas.toBlob(
          (blob) => (blob ? resolve(blob) : reject(new Error("Could not process the photo."))),
          "image/jpeg",
          JPEG_QUALITY
        );
      };

      img.onerror = () => {
        URL.revokeObjectURL(url);
        reject(new Error("Could not read the photo."));
      };

      img.src = url;
    });
  }

  async function checkDisease(file, crop) {
    const apiUrl = Config.get("apiUrl");
    const compressed = await _compressImage(file);

    const formData = new FormData();
    formData.append("image", compressed, "leaf.jpg");
    formData.append("crop", crop);
    formData.append("language", LanguageSelector.getLanguage());

    const response = await fetch(`${apiUrl}/api/disease`, {
      method: "POST",
      body: formData,
    });

    const data = await response.json().catch(() => ({}));

    if (!response.ok) {
      // disease_api's own 400/413 errors come back as {"error": ...}; backend-raised
      // errors (no crop, service down) come back as {"detail": ...} — check both.
      const err = new Error(data.error || data.detail || `API Error: ${response.status}`);
      err.status = response.status;
      throw err;
    }

    return data;
  }

  return { checkDisease };
})();


/* ─── Chat History ───────────────────────────────────────── */
const ChatHistory = (() => {
  let currentChatId = null;

  async function ensureChatId(title) {
    if (!currentChatId) {
      try {
        const chat = await API.createChat(title);
        currentChatId = chat.chat_id;
        await loadSidebar();
      } catch (err) {
        console.error("Failed to create chat", err);
      }
    }
    return currentChatId;
  }

  async function loadSidebar() {
    try {
      const result = await API.getChats();
      const list = document.getElementById("chat-list");
      list.innerHTML = "";
      for (const chat of result.chats) {
        const li = document.createElement("li");
        li.className = "chat-item" + (chat.id === currentChatId ? " active" : "");
        li.textContent = chat.title;
        li.addEventListener("click", () => loadChat(chat.id));
        list.appendChild(li);
      }
    } catch (err) {
      console.error("Failed to load sidebar", err);
    }
  }

  async function loadChat(chatId) {
    currentChatId = chatId;
    document.getElementById("chat-area").innerHTML = "";
    
    // Restore welcome screen just to hide it cleanly when messages load
    const welcome = document.getElementById("welcome-screen");
    if (welcome) welcome.classList.add("hidden");
    
    await loadSidebar(); // update active class
    
    // on mobile, close sidebar
    document.getElementById("history-sidebar").classList.remove("open");
    
    try {
      const result = await API.getChatMessages(chatId);
      for (const msg of result.messages) {
        if (msg.msg_type === "speech_result") {
          ChatUI.addSpeechResult(msg.content.marathi_text, msg.content.english_text);
        } else if (msg.msg_type === "user_text") {
          ChatUI.addUserText(msg.content.english_text);
        } else if (msg.msg_type === "bot_tts") {
          // Older chats stored an English reply separately from the Marathi TTS text
          if (msg.content.ai_response && !msg.content.reply_language) {
            ChatUI.addBotText("AI Response (English)", msg.content.ai_response);
          }

          ChatUI.addTTSResult(
            msg.content.marathi_text,
            msg.content.word_timings,
            msg.content.audio_base64,
            msg.content.voice,
            msg.content.duration
          );
        } else if (msg.msg_type === "bot_text") {
          ChatUI.addBotText("Voice not available", msg.content.ai_response);
        }
      }
    } catch (err) {
      console.error("Failed to load messages", err);
    }
  }

  function startNewChat() {
    currentChatId = null;
    document.getElementById("chat-area").innerHTML = "";
    
    // Add welcome back if exists, or recreate it
    let welcome = document.getElementById("welcome-screen");
    if (!welcome) {
      welcome = document.createElement("div");
      welcome.className = "welcome-screen";
      welcome.id = "welcome-screen";
      welcome.innerHTML = `
        <div class="welcome-icon" aria-hidden="true">🌾</div>
        <div>
          <h1 class="welcome-title">नमस्कार! Speak <span>Marathi</span>,<br />Hear it in <span>English</span></h1>
          <p class="welcome-subtitle">Hold the mic button and speak in Marathi.<br />Or type English text to hear it spoken in Marathi.</p>
        </div>
        <div class="welcome-tips" role="list">
          <div class="tip-card" role="listitem"><span class="tip-icon">🎙️</span><span>Hold mic &amp; speak <strong>Marathi</strong></span></div>
          <div class="tip-card" role="listitem"><span class="tip-icon">📝</span><span>Type <strong>English</strong> text</span></div>
          <div class="tip-card" role="listitem"><span class="tip-icon">🔊</span><span>Tap ↺ to <strong>replay</strong> Marathi speech</span></div>
        </div>
      `;
      document.getElementById("chat-area").appendChild(welcome);
    } else {
      welcome.classList.remove("hidden");
      document.getElementById("chat-area").appendChild(welcome);
    }
    
    loadSidebar();
    document.getElementById("history-sidebar").classList.remove("open");
  }

  function init() {
    document.getElementById("new-chat-btn").addEventListener("click", startNewChat);
    document.getElementById("sidebar-toggle-btn").addEventListener("click", () => {
      document.getElementById("history-sidebar").classList.toggle("open");
    });
    document.getElementById("sidebar-close-btn").addEventListener("click", () => {
      document.getElementById("history-sidebar").classList.remove("open");
    });
    loadSidebar();
  }

  function getChatId() { return currentChatId; }

  return { init, ensureChatId, getChatId };
})();


/* ─── Toast Notifications ────────────────────────────────── */
const Toast = (() => {
  let container;

  function init() {
    container = document.createElement("div");
    container.className = "toast-container";
    document.body.appendChild(container);
  }

  function show(message, type = "info", duration = 3000) {
    if (!container) init();
    const toast = document.createElement("div");
    toast.className = `toast ${type}`;
    toast.textContent = message;
    container.appendChild(toast);
    setTimeout(() => toast.remove(), duration);
  }

  return { show };
})();


/* ─── TTS Player ─────────────────────────────────────────── */
const TTSPlayer = (() => {
  let _audioCtx = null;
  let _sourceNode = null;
  let _audioBuffer = null;
  let _wordTimings = [];
  let _startTime = 0;
  let _pausedAt = 0;
  let _isPlaying = false;
  let _highlightInterval = null;
  let _onWordChange = null;
  let _onProgress = null;
  let _onEnd = null;

  function _getAudioCtx() {
    if (!_audioCtx || _audioCtx.state === "closed") {
      _audioCtx = new (window.AudioContext || window.webkitAudioContext)();
    }
    return _audioCtx;
  }

  async function load(base64Audio, wordTimings, callbacks = {}) {
    stop();
    _wordTimings = wordTimings || [];
    _onWordChange = callbacks.onWordChange || null;
    _onProgress = callbacks.onProgress || null;
    _onEnd = callbacks.onEnd || null;
    _pausedAt = 0;

    const ctx = _getAudioCtx();
    const bytes = Uint8Array.from(atob(base64Audio), (c) => c.charCodeAt(0));
    _audioBuffer = await ctx.decodeAudioData(bytes.buffer);

    // Scale word timings to match exact decoded audio duration
    if (_wordTimings.length > 0 && _audioBuffer && _audioBuffer.duration > 0) {
      const actualDuration = _audioBuffer.duration;
      // Get the backend's estimated duration from the last word's end time, or 1.0
      const estimatedDuration = _wordTimings[_wordTimings.length - 1].end || 1.0;
      const scale = actualDuration / estimatedDuration;

      _wordTimings = _wordTimings.map(w => ({
        word: w.word,
        start: w.start * scale,
        end: w.end * scale
      }));
    }
  }

  function _startHighlighting(offsetTime = 0) {
    if (_highlightInterval) clearInterval(_highlightInterval);

    _highlightInterval = setInterval(() => {
      if (!_isPlaying) return;

      const ctx = _audioCtx;
      const elapsed = ctx.currentTime - _startTime + offsetTime;
      const total = _audioBuffer ? _audioBuffer.duration : 1;
      const progress = Math.min((elapsed / total) * 100, 100);

      if (_onProgress) _onProgress(progress, elapsed, total);

      // Find current word
      let activeIdx = -1;
      for (let i = 0; i < _wordTimings.length; i++) {
        const w = _wordTimings[i];
        if (elapsed >= w.start && elapsed < w.end) {
          activeIdx = i;
          break;
        }
      }
      if (_onWordChange) _onWordChange(activeIdx, elapsed);

      if (elapsed >= total) {
        _finishPlayback();
      }
    }, 50);
  }

  function _finishPlayback() {
    _isPlaying = false;
    if (_highlightInterval) clearInterval(_highlightInterval);
    if (_onEnd) _onEnd();
    if (_onWordChange) _onWordChange(-1, 0);
    if (_onProgress) _onProgress(0, 0, 1);
    _pausedAt = 0;
  }

  function play(fromOffset = 0) {
    if (!_audioBuffer) return;
    stop();

    const ctx = _getAudioCtx();
    if (ctx.state === "suspended") ctx.resume();

    _sourceNode = ctx.createBufferSource();
    _sourceNode.buffer = _audioBuffer;
    _sourceNode.connect(ctx.destination);
    _sourceNode.start(0, fromOffset);
    _sourceNode.onended = () => {
      if (_isPlaying) _finishPlayback();
    };

    _startTime = ctx.currentTime - fromOffset;
    _isPlaying = true;
    _startHighlighting(fromOffset);
  }

  function stop() {
    if (_sourceNode) {
      try {
        _sourceNode.stop();
        _sourceNode.disconnect();
      } catch (e) { /* ignore */ }
      _sourceNode = null;
    }
    _isPlaying = false;
    if (_highlightInterval) {
      clearInterval(_highlightInterval);
      _highlightInterval = null;
    }
  }

  function pause() {
    if (!_isPlaying || !_audioCtx) return;
    _pausedAt = _audioCtx.currentTime - _startTime;
    stop();
  }

  function resume() {
    play(_pausedAt);
  }

  function replay() {
    _pausedAt = 0;
    play(0);
  }

  function isPlaying() { return _isPlaying; }
  function isPaused() { return !_isPlaying && _pausedAt > 0; }

  return { load, play, stop, pause, resume, replay, isPlaying, isPaused };
})();


/* ─── Speech Input (Mic Recording + Web Speech API) ──────── */
const SpeechInput = (() => {
  let _mediaRecorder = null;
  let _audioChunks = [];
  let _recognition = null;
  let _isRecording = false;
  let _onInterim = null;    // live Marathi text callback
  let _onFinal = null;      // final audio blob callback
  let _onError = null;

  // Setup Web Speech API for live Marathi transcription (Chrome only)
  function _initSpeechRecognition() {
    const SpeechRecognition =
      window.SpeechRecognition || window.webkitSpeechRecognition;

    if (!SpeechRecognition) return null;

    const recog = new SpeechRecognition();
    recog.lang = "mr-IN";        // Marathi India
    recog.continuous = true;
    recog.interimResults = true;
    recog.maxAlternatives = 1;

    recog.onresult = (event) => {
      let interimText = "";
      let finalText = "";
      for (let i = event.resultIndex; i < event.results.length; i++) {
        const t = event.results[i][0].transcript;
        if (event.results[i].isFinal) {
          finalText += t;
        } else {
          interimText += t;
        }
      }
      if (_onInterim) _onInterim(finalText + interimText, finalText, interimText);
    };

    recog.onerror = (e) => {
      if (e.error !== "no-speech") {
        console.warn("Speech recognition error:", e.error);
      }
    };

    return recog;
  }

  async function _startMediaRecorder(stream) {
    // Prefer audio/webm;codecs=opus (best quality), fall back to audio/ogg or audio/wav
    const mimeTypes = [
      "audio/webm;codecs=opus",
      "audio/webm",
      "audio/ogg;codecs=opus",
      "audio/ogg",
      "audio/wav",
    ];
    let mimeType = "";
    for (const mt of mimeTypes) {
      if (MediaRecorder.isTypeSupported(mt)) {
        mimeType = mt;
        break;
      }
    }

    _audioChunks = [];
    const options = mimeType ? { mimeType } : {};
    _mediaRecorder = new MediaRecorder(stream, options);

    _mediaRecorder.ondataavailable = (e) => {
      if (e.data.size > 0) _audioChunks.push(e.data);
    };

    _mediaRecorder.onstop = () => {
      const blob = new Blob(_audioChunks, { type: mimeType || "audio/webm" });
      if (_onFinal) _onFinal(blob);
    };

    _mediaRecorder.start(100); // collect chunks every 100ms
  }

  async function startRecording(callbacks = {}) {
    if (_isRecording) return;
    _onInterim = callbacks.onInterim || null;
    _onFinal = callbacks.onFinal || null;
    _onError = callbacks.onError || null;

    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        audio: {
          channelCount: 1,
          sampleRate: 16000,
          echoCancellation: true,
          noiseSuppression: true,
        },
      });

      await _startMediaRecorder(stream);
      _isRecording = true;

      // Start live speech recognition in parallel
      _recognition = _initSpeechRecognition();
      if (_recognition) {
        _recognition.start();
      }

      return stream;
    } catch (err) {
      console.error("Mic access error:", err);
      if (_onError) _onError(err.name === "NotAllowedError"
        ? "Microphone permission denied. Please allow mic access."
        : "Could not access microphone: " + err.message
      );
      return null;
    }
  }

  function stopRecording() {
    if (!_isRecording) return;
    _isRecording = false;

    if (_recognition) {
      try { _recognition.stop(); } catch (e) { /* ignore */ }
      _recognition = null;
    }

    if (_mediaRecorder && _mediaRecorder.state !== "inactive") {
      _mediaRecorder.stop();
      // Stop mic stream
      _mediaRecorder.stream.getTracks().forEach((t) => t.stop());
    }
  }

  function isRecording() { return _isRecording; }
  function hasWebSpeech() {
    return !!(window.SpeechRecognition || window.webkitSpeechRecognition);
  }

  return { startRecording, stopRecording, isRecording, hasWebSpeech };
})();


/* ─── Chat UI ─────────────────────────────────────────────── */
const ChatUI = (() => {
  let _chatArea;
  let _welcomeScreen;

  function init(chatArea, welcomeScreen) {
    _chatArea = chatArea;
    _welcomeScreen = welcomeScreen;
  }

  function _hideWelcome() {
    if (_welcomeScreen && !_welcomeScreen.classList.contains("hidden")) {
      _welcomeScreen.classList.add("hidden");
    }
  }

  function _scrollToBottom() {
    requestAnimationFrame(() => {
      _chatArea.scrollTop = _chatArea.scrollHeight;
    });
  }

  function _createAvatar(type) {
    const el = document.createElement("div");
    el.className = "message-avatar";
    el.textContent = type === "user" ? "🧑" : "🤖";
    return el;
  }

  function _createLabel(text) {
    const el = document.createElement("div");
    el.className = "message-label";
    el.textContent = text;
    return el;
  }

  /**
   * Add a user Marathi speech message (2 bubbles: Marathi + English)
   */
  function addSpeechResult(marathiText, englishText) {
    _hideWelcome();

    // User side: Marathi speech bubble
    const userMsg = document.createElement("div");
    userMsg.className = "message user";
    const userBody = document.createElement("div");
    userBody.className = "message-body";
    userBody.appendChild(_createLabel("आपण बोललात (Marathi)"));
    const marathiBubble = document.createElement("div");
    marathiBubble.className = "bubble marathi-speech";
    marathiBubble.textContent = marathiText;
    userBody.appendChild(marathiBubble);
    userMsg.appendChild(_createAvatar("user"));
    userMsg.appendChild(userBody);
    _chatArea.appendChild(userMsg);

    if (!englishText) {
      _scrollToBottom();
      return { userMsg, botMsg: null };
    }

    // Bot side: English translation bubble
    const botMsg = document.createElement("div");
    botMsg.className = "message bot";
    const botBody = document.createElement("div");
    botBody.className = "message-body";
    botBody.appendChild(_createLabel("English Translation"));
    const englishBubble = document.createElement("div");
    englishBubble.className = "bubble english-translation";
    englishBubble.textContent = englishText;
    botBody.appendChild(englishBubble);
    botMsg.appendChild(_createAvatar("bot"));
    botMsg.appendChild(botBody);
    _chatArea.appendChild(botMsg);

    _scrollToBottom();
    return { userMsg, botMsg };
  }

  /**
   * Add a plain bot text bubble (textContent — LLM output is never parsed as HTML)
   */
  function addBotText(label, text) {
    _hideWelcome();
    const msg = document.createElement("div");
    msg.className = "message bot";
    const body = document.createElement("div");
    body.className = "message-body";
    body.appendChild(_createLabel(label));
    const bubble = document.createElement("div");
    bubble.className = "bubble english-translation";
    bubble.textContent = text;
    body.appendChild(bubble);
    msg.appendChild(_createAvatar("bot"));
    msg.appendChild(body);
    _chatArea.appendChild(msg);
    _scrollToBottom();
    return msg;
  }

  /**
   * Add user English text message
   */
  function addUserText(englishText) {
    _hideWelcome();
    const msg = document.createElement("div");
    msg.className = "message user";
    const body = document.createElement("div");
    body.className = "message-body";
    body.appendChild(_createLabel("English Text"));
    const bubble = document.createElement("div");
    bubble.className = "bubble user-text";
    bubble.textContent = englishText;
    body.appendChild(bubble);
    msg.appendChild(_createAvatar("user"));
    msg.appendChild(body);
    _chatArea.appendChild(msg);
    _scrollToBottom();
    return msg;
  }

  /**
   * Add loading typing bubble (returns element to remove later)
   */
  function addTypingBubble() {
    _hideWelcome();
    const msg = document.createElement("div");
    msg.className = "message bot";
    const body = document.createElement("div");
    body.className = "message-body";
    const typing = document.createElement("div");
    typing.className = "typing-bubble";
    [1, 2, 3].forEach(() => {
      const dot = document.createElement("div");
      dot.className = "typing-dot";
      typing.appendChild(dot);
    });
    body.appendChild(typing);
    msg.appendChild(_createAvatar("bot"));
    msg.appendChild(body);
    _chatArea.appendChild(msg);
    _scrollToBottom();
    return msg;
  }

  /**
   * Add TTS result bubble with word highlighting + player controls
   */
  function addTTSResult(marathiText, wordTimings, audioBase64, voice, duration) {
    const msg = document.createElement("div");
    msg.className = "message bot";
    const body = document.createElement("div");
    body.className = "message-body";
    body.appendChild(_createLabel(`🔊 Marathi Speech (${voice})`));

    const bubble = document.createElement("div");
    bubble.className = "bubble tts-bubble";

    // Word spans
    const wordsDiv = document.createElement("div");
    wordsDiv.className = "tts-words";

    const wordSpans = wordTimings.map((w, i) => {
      const span = document.createElement("span");
      span.className = "tts-word";
      span.dataset.index = i;
      span.textContent = w.word + " ";
      wordsDiv.appendChild(span);
      return span;
    });

    // If no word timings, just show plain text
    if (wordTimings.length === 0) {
      wordsDiv.textContent = marathiText;
      wordsDiv.style.fontFamily = "var(--font)"; // Mukta, from style.css tokens
    }

    bubble.appendChild(wordsDiv);

    // Player controls
    const player = document.createElement("div");
    player.className = "tts-player";

    const controls = document.createElement("div");
    controls.className = "tts-controls";

    // Play/Pause button
    const playBtn = document.createElement("button");
    playBtn.className = "tts-btn";
    playBtn.id = `play-${Date.now()}`;
    playBtn.title = "Play";
    playBtn.innerHTML = "▶";
    playBtn.setAttribute("aria-label", "Play Marathi speech");

    // Replay button
    const replayBtn = document.createElement("button");
    replayBtn.className = "tts-btn";
    replayBtn.title = "Replay from start";
    replayBtn.innerHTML = "↺";
    replayBtn.setAttribute("aria-label", "Replay Marathi speech");

    // Progress bar
    const progressBar = document.createElement("div");
    progressBar.className = "tts-progress";
    const progressFill = document.createElement("div");
    progressFill.className = "tts-progress-fill";
    progressBar.appendChild(progressFill);

    // Time display
    const timeEl = document.createElement("div");
    timeEl.className = "tts-time";
    timeEl.textContent = `0:00 / ${_formatTime(duration)}`;

    // Voice badge
    const voiceBadge = document.createElement("div");
    voiceBadge.className = "tts-voice-badge";
    voiceBadge.textContent = voice;

    controls.appendChild(playBtn);
    controls.appendChild(replayBtn);
    controls.appendChild(progressBar);
    controls.appendChild(timeEl);
    controls.appendChild(voiceBadge);
    player.appendChild(controls);
    bubble.appendChild(player);
    body.appendChild(bubble);

    msg.appendChild(_createAvatar("bot"));
    msg.appendChild(body);
    _chatArea.appendChild(msg);
    _scrollToBottom();

    // ── Wire up TTS player ──────────────────────────────────
    let playerLoaded = false;

    async function _ensureLoaded() {
      if (playerLoaded) return;
      playerLoaded = true;
      await TTSPlayer.load(audioBase64, wordTimings, {
        onWordChange: (activeIdx) => {
          wordSpans.forEach((span, i) => {
            span.classList.toggle("active", i === activeIdx);
            span.classList.toggle("spoken", activeIdx >= 0 && i < activeIdx);
          });
        },
        onProgress: (pct, elapsed, total) => {
          progressFill.style.width = pct + "%";
          timeEl.textContent = `${_formatTime(elapsed)} / ${_formatTime(total)}`;
        },
        onEnd: () => {
          playBtn.innerHTML = "▶";
          playBtn.classList.remove("playing");
          wordSpans.forEach((s) => s.classList.remove("active", "spoken"));
          progressFill.style.width = "0%";
          timeEl.textContent = `0:00 / ${_formatTime(duration)}`;
        },
      });
    }

    playBtn.addEventListener("click", async () => {
      await _ensureLoaded();
      if (TTSPlayer.isPlaying()) {
        TTSPlayer.pause();
        playBtn.innerHTML = "▶";
        playBtn.classList.remove("playing");
      } else if (TTSPlayer.isPaused()) {
        TTSPlayer.resume();
        playBtn.innerHTML = "⏸";
        playBtn.classList.add("playing");
      } else {
        TTSPlayer.play(0);
        playBtn.innerHTML = "⏸";
        playBtn.classList.add("playing");
      }
    });

    replayBtn.addEventListener("click", async () => {
      await _ensureLoaded();
      TTSPlayer.replay();
      playBtn.innerHTML = "⏸";
      playBtn.classList.add("playing");
    });

    return msg;
  }

  /**
   * Add a user leaf-photo thumbnail bubble
   */
  function addUserPhoto(thumbnailUrl, crop) {
    _hideWelcome();
    const msg = document.createElement("div");
    msg.className = "message user";
    const body = document.createElement("div");
    body.className = "message-body";
    body.appendChild(_createLabel(`Leaf Photo (${crop})`));
    const bubble = document.createElement("div");
    bubble.className = "bubble photo-bubble";
    const img = document.createElement("img");
    img.className = "photo-thumbnail";
    img.src = thumbnailUrl;
    img.alt = `${crop} leaf photo`;
    img.addEventListener("load", () => URL.revokeObjectURL(thumbnailUrl));
    bubble.appendChild(img);
    body.appendChild(bubble);
    msg.appendChild(_createAvatar("user"));
    msg.appendChild(body);
    _chatArea.appendChild(msg);
    _scrollToBottom();
    return msg;
  }

  /**
   * Add a "Checking your leaf..." loading bubble (bot side, returns element to remove later)
   */
  function addDiseaseLoading(text) {
    _hideWelcome();
    const msg = document.createElement("div");
    msg.className = "message bot";
    const body = document.createElement("div");
    body.className = "message-body";
    const bubble = document.createElement("div");
    bubble.className = "bubble english-translation disease-loading";
    const label = document.createElement("span");
    label.textContent = text;
    bubble.appendChild(label);
    const dotRow = document.createElement("div");
    dotRow.className = "typing-dot-row";
    [1, 2, 3].forEach(() => {
      const dot = document.createElement("div");
      dot.className = "typing-dot";
      dotRow.appendChild(dot);
    });
    bubble.appendChild(dotRow);
    body.appendChild(bubble);
    msg.appendChild(_createAvatar("bot"));
    msg.appendChild(body);
    _chatArea.appendChild(msg);
    _scrollToBottom();
    return msg;
  }

  /**
   * Add the disease-check result bubble (bot side).
   * "ok": message + both guesses as "name (English name) — NN%".
   * "not_sure": only the message, no guesses.
   */
  function addDiseaseResult(result) {
    _hideWelcome();
    const msg = document.createElement("div");
    msg.className = "message bot";
    const body = document.createElement("div");
    body.className = "message-body";
    body.appendChild(_createLabel("🌿 Leaf Check Result"));

    const bubble = document.createElement("div");
    bubble.className = "bubble disease-result";

    const messageEl = document.createElement("div");
    messageEl.className = "disease-message";
    messageEl.textContent = result.message;
    bubble.appendChild(messageEl);

    if (result.status === "ok" && Array.isArray(result.top) && result.top.length > 0) {
      const guesses = document.createElement("div");
      guesses.className = "disease-guesses";
      result.top.forEach((g) => {
        const row = document.createElement("div");
        row.className = "disease-guess-row";
        const pct = Math.round((g.probability || 0) * 100);
        row.textContent = `${g.disease} (${g.disease_en}) — ${pct}%`;
        guesses.appendChild(row);
      });
      bubble.appendChild(guesses);
    }

    if (Array.isArray(result.precautions) && result.precautions.length > 0) {
      const precautionsEl = document.createElement("div");
      precautionsEl.className = "disease-precautions";

      const title = document.createElement("div");
      title.className = "disease-precautions-title";
      title.textContent = result.precautions_title || "Precautions";
      precautionsEl.appendChild(title);

      const list = document.createElement("ul");
      list.className = "disease-precautions-list";
      result.precautions.forEach((line) => {
        const li = document.createElement("li");
        li.textContent = line;
        list.appendChild(li);
      });
      precautionsEl.appendChild(list);

      bubble.appendChild(precautionsEl);
    }

    body.appendChild(bubble);
    msg.appendChild(_createAvatar("bot"));
    msg.appendChild(body);
    _chatArea.appendChild(msg);
    _scrollToBottom();
    return msg;
  }

  /**
   * Add an error bubble (bot side)
   */
  function addError(message) {
    const msg = document.createElement("div");
    msg.className = "message bot";
    const body = document.createElement("div");
    body.className = "message-body";
    const bubble = document.createElement("div");
    bubble.className = "bubble error";
    bubble.textContent = "⚠️ " + message;
    body.appendChild(bubble);
    msg.appendChild(_createAvatar("bot"));
    msg.appendChild(body);
    _chatArea.appendChild(msg);
    _scrollToBottom();
    return msg;
  }

  /**
   * Add the Crop Advisor result bubble (bot side).
   * `labels` is CropI18n's label object for the farmer's current language
   * (resultLabel, riskLabel, sourceLabel, rainLabel, mm, monthName()).
   * `onSpeak` is called (with no args) when the 🔊 button is tapped —
   * speaking is on-tap only, never automatic, since this text is long.
   */
  function addCropResult(result, labels, onSpeak) {
    _hideWelcome();
    const msg = document.createElement("div");
    msg.className = "message bot";
    const body = document.createElement("div");
    body.className = "message-body";
    body.appendChild(_createLabel(labels.resultLabel));

    const bubble = document.createElement("div");
    bubble.className = "bubble crop-result";

    const speakRow = document.createElement("div");
    speakRow.className = "crop-result-speak-row";
    const speakBtn = document.createElement("button");
    speakBtn.className = "crop-result-speak-btn";
    speakBtn.type = "button";
    speakBtn.innerHTML = "🔊";
    speakBtn.title = labels.speakTitle;
    speakBtn.setAttribute("aria-label", labels.speakTitle);
    speakBtn.addEventListener("click", () => {
      if (onSpeak) onSpeak(speakBtn);
    });
    speakRow.appendChild(speakBtn);
    bubble.appendChild(speakRow);

    (result.crops || []).forEach((crop) => {
      const item = document.createElement("div");
      item.className = "crop-result-item";

      const nameEl = document.createElement("div");
      nameEl.className = "crop-result-name";
      nameEl.textContent = crop.name;
      item.appendChild(nameEl);

      if (Array.isArray(crop.sowing_months) && crop.sowing_months.length > 0) {
        const monthsEl = document.createElement("div");
        monthsEl.className = "crop-result-months";
        monthsEl.textContent = crop.sowing_months.map(labels.monthName).join(", ");
        item.appendChild(monthsEl);
      }

      if (crop.reason) {
        const reasonEl = document.createElement("div");
        reasonEl.className = "crop-result-reason";
        reasonEl.textContent = crop.reason;
        item.appendChild(reasonEl);
      }

      if (crop.risk) {
        const riskEl = document.createElement("div");
        riskEl.className = "crop-result-risk";
        const riskLabel = document.createElement("span");
        riskLabel.className = "crop-result-risk-label";
        riskLabel.textContent = labels.riskLabel + ": ";
        riskEl.appendChild(riskLabel);
        riskEl.appendChild(document.createTextNode(crop.risk));
        item.appendChild(riskEl);
      }

      if (Array.isArray(crop.evidence_refs) && crop.evidence_refs.length > 0) {
        const sourceEl = document.createElement("div");
        sourceEl.className = "crop-result-source";
        const refs = crop.evidence_refs
          .map((r) => `p${r.page}${r.section ? " sec " + r.section : ""}`)
          .join(", ");
        sourceEl.textContent = `${labels.sourceLabel}: ICAR 2011, ${refs}`;
        item.appendChild(sourceEl);
      }

      bubble.appendChild(item);
    });

    if (result.weather && typeof result.weather.rain_last_14_days_mm === "number") {
      const weatherEl = document.createElement("div");
      weatherEl.className = "crop-result-weather";
      weatherEl.textContent = `${labels.rainLabel}: ${result.weather.rain_last_14_days_mm} ${labels.mm}`;
      bubble.appendChild(weatherEl);
    }

    if (result.disclaimer) {
      const disclaimerEl = document.createElement("div");
      disclaimerEl.className = "crop-result-disclaimer";
      disclaimerEl.textContent = result.disclaimer;
      bubble.appendChild(disclaimerEl);
    }

    body.appendChild(bubble);
    msg.appendChild(_createAvatar("bot"));
    msg.appendChild(body);
    _chatArea.appendChild(msg);
    _scrollToBottom();
    return { msg, speakBtn };
  }

  function _formatTime(seconds) {
    if (!isFinite(seconds) || isNaN(seconds)) return "0:00";
    const m = Math.floor(seconds / 60);
    const s = Math.floor(seconds % 60).toString().padStart(2, "0");
    return `${m}:${s}`;
  }

  return {
    init,
    addSpeechResult,
    addBotText,
    addUserText,
    addTypingBubble,
    addTTSResult,
    addUserPhoto,
    addDiseaseLoading,
    addDiseaseResult,
    addCropResult,
    addError,
  };
})();


/* ─── Settings Modal ─────────────────────────────────────── */
const Settings = (() => {
  let _modal, _overlay;
  const VOICES = ["Sunita", "Sanjay", "Nikhil", "Radha", "Varun", "Isha"];
  let _selectedVoice = Config.get("voice");

  function init() {
    _overlay = document.getElementById("settings-overlay");
    _modal = document.getElementById("settings-modal");

    // Build voice grid
    const voiceGrid = document.getElementById("voice-grid");
    VOICES.forEach((voice) => {
      const btn = document.createElement("button");
      btn.className = `voice-btn${voice === _selectedVoice ? " selected" : ""}`;
      btn.textContent = voice;
      btn.dataset.voice = voice;
      btn.addEventListener("click", () => {
        voiceGrid.querySelectorAll(".voice-btn").forEach((b) => b.classList.remove("selected"));
        btn.classList.add("selected");
        _selectedVoice = voice;
      });
      voiceGrid.appendChild(btn);
    });

    // Load saved values
    document.getElementById("api-url-input").value = Config.get("apiUrl");

    // Save button
    document.getElementById("settings-save").addEventListener("click", () => {
      // Empty box = go back to the address in config.js. Drop a trailing "/"
      // so we never call "...//api/chat".
      const typed = document.getElementById("api-url-input").value.trim().replace(/\/+$/, "");
      const apiUrl = typed || Config.defaultApiUrl();
      Config.save({ apiUrl, voice: _selectedVoice });
      close();
      Toast.show(`Settings saved! Voice: ${_selectedVoice}`, "success");
    });

    // Close
    document.getElementById("settings-close").addEventListener("click", close);
    _overlay.addEventListener("click", (e) => {
      if (e.target === _overlay) close();
    });
  }

  function open() {
    _overlay.classList.add("open");
    document.getElementById("api-url-input").value = Config.get("apiUrl");
  }

  function close() {
    _overlay.classList.remove("open");
  }

  return { init, open, close };
})();


/* ─── Crop Picker Modal (leaf disease check) ─────────────── */
const CropPicker = (() => {
  let _overlay;
  let _selectedCrop = null;
  const CROPS = [
    { name: "Cotton", icon: "🌱" },
    { name: "Grape", icon: "🍇" },
    { name: "Maize", icon: "🌽" },
    { name: "Onion", icon: "🧅" },
    { name: "Tomato", icon: "🍅" },
  ];

  function init(onCropChosen) {
    _overlay = document.getElementById("crop-overlay");

    const cropGrid = document.getElementById("crop-grid");
    CROPS.forEach(({ name, icon }) => {
      const btn = document.createElement("button");
      btn.className = "crop-btn";
      btn.dataset.crop = name;
      btn.innerHTML = `<span class="crop-icon" aria-hidden="true">${icon}</span><span>${name}</span>`;
      btn.addEventListener("click", () => {
        _selectedCrop = name;
        close();
        if (onCropChosen) onCropChosen(name);
      });
      cropGrid.appendChild(btn);
    });

    document.getElementById("crop-close").addEventListener("click", close);
    _overlay.addEventListener("click", (e) => {
      if (e.target === _overlay) close();
    });
  }

  function open() {
    _overlay.classList.add("open");
  }

  function close() {
    _overlay.classList.remove("open");
  }

  function getSelectedCrop() { return _selectedCrop; }

  return { init, open, close, getSelectedCrop };
})();


/* ─── Crop Advisor i18n (Module 2 Phase 5) ───────────────────
   The app has no other UI-string localization yet (LanguageSelector only
   picks the backend's reply language) — this is the first one, built
   fresh for the Crop Advisor form and result card. reviewed:false: a
   first-pass mr/hi translation, not checked by a native speaker/KVK yet
   (same convention as backend/data/crop_messages.json). ─────────────── */
const CropI18n = (() => {
  const STRINGS = {
    buttonLabel:  { mr: "पीक सल्ला", hi: "फसल सलाह", en: "Crop Advisor" },
    modalTitle:   { mr: "🌾 पीक सल्ला", hi: "🌾 फसल सलाह", en: "🌾 Crop Advisor" },
    taluka:       { mr: "तालुका", hi: "तालुका", en: "Taluka" },
    sowingMonth:  { mr: "पेरणीचा महिना", hi: "बुवाई का महीना", en: "Sowing month" },
    irrigation:   { mr: "पाण्याची सुविधा आहे का?", hi: "सिंचाई की सुविधा है?", en: "Irrigation available?" },
    yes:          { mr: "हो", hi: "हाँ", en: "Yes" },
    no:           { mr: "नाही", hi: "नहीं", en: "No" },
    soilType:     { mr: "जमिनीचा प्रकार", hi: "मिट्टी का प्रकार", en: "Soil type" },
    previousCrop: { mr: "मागील पीक", hi: "पिछली फसल", en: "Previous crop" },
    none:         { mr: "काहीही नाही", hi: "कोई नहीं", en: "None" },
    notSure:      { mr: "माहीत नाही", hi: "पता नहीं", en: "Not sure" },
    soilCard:     { mr: "माती आरोग्य पत्रिका मूल्ये (पर्यायी)", hi: "मृदा स्वास्थ्य कार्ड मूल्य (वैकल्पिक)", en: "Soil Health Card values (optional)" },
    ph:           { mr: "pH", hi: "pH", en: "pH" },
    ec:           { mr: "EC (dS/m)", hi: "EC (dS/m)", en: "EC (dS/m)" },
    oc:           { mr: "OC (%)", hi: "OC (%)", en: "OC (%)" },
    n:            { mr: "N (kg/ha)", hi: "N (kg/ha)", en: "N (kg/ha)" },
    p:            { mr: "P (kg/ha)", hi: "P (kg/ha)", en: "P (kg/ha)" },
    k:            { mr: "K (kg/ha)", hi: "K (kg/ha)", en: "K (kg/ha)" },
    submit:       { mr: "पीक सल्ला मिळवा", hi: "फसल सलाह पाएं", en: "Get crop advice" },
    loading:      { mr: "पीक सल्ला तयार होत आहे…", hi: "फसल सलाह तैयार हो रही है…", en: "Preparing crop advice…" },
    timeoutError: { mr: "सध्या उत्तर मिळत नाही. कृपया पुन्हा प्रयत्न करा.", hi: "अभी जवाब नहीं मिल रहा है। कृपया फिर से प्रयास करें.", en: "This is taking too long. Please try again." },
    optionsError: { mr: "फॉर्मसाठी माहिती मिळाली नाही. कृपया नंतर प्रयत्न करा.", hi: "फ़ॉर्म के लिए जानकारी नहीं मिली। कृपया बाद में प्रयास करें.", en: "Could not load the form. Please try again later." },
    resultLabel:  { mr: "🌾 पीक सल्ला निकाल", hi: "🌾 फसल सलाह परिणाम", en: "🌾 Crop Advisor Result" },
    riskLabel:    { mr: "धोका", hi: "जोखिम", en: "Risk" },
    sourceLabel:  { mr: "स्रोत", hi: "स्रोत", en: "Source" },
    rainLabel:    { mr: "गेल्या १४ दिवसांतील पाऊस", hi: "पिछले 14 दिनों की बारिश", en: "Rain in the last 14 days" },
    mm:           { mr: "मिमी", hi: "मिमी", en: "mm" },
    speakTitle:   { mr: "ऐकण्यासाठी दाबा", hi: "सुनने के लिए दबाएँ", en: "Tap to hear this" },
  };

  const MONTHS = {
    mr: ["जानेवारी", "फेब्रुवारी", "मार्च", "एप्रिल", "मे", "जून", "जुलै", "ऑगस्ट", "सप्टेंबर", "ऑक्टोबर", "नोव्हेंबर", "डिसेंबर"],
    hi: ["जनवरी", "फरवरी", "मार्च", "अप्रैल", "मई", "जून", "जुलाई", "अगस्त", "सितंबर", "अक्टूबर", "नवंबर", "दिसंबर"],
    en: ["January", "February", "March", "April", "May", "June", "July", "August", "September", "October", "November", "December"],
  };

  function t(key, lang) {
    const entry = STRINGS[key];
    if (!entry) return key;
    return entry[lang] || entry.en || key;
  }

  function monthName(monthNum, lang) {
    const arr = MONTHS[lang] || MONTHS.en;
    return arr[monthNum - 1] || String(monthNum);
  }

  // A snapshot of every label for one language, so call sites (e.g.
  // ChatUI.addCropResult) don't each need to know the current language.
  function getLabels(lang) {
    const labels = {};
    Object.keys(STRINGS).forEach((key) => { labels[key] = t(key, lang); });
    labels.monthName = (monthNum) => monthName(monthNum, lang);
    return labels;
  }

  return { t, monthName, getLabels };
})();


/* ─── Crop Advisor Modal (Module 2 Phase 5) ──────────────────
   Form UI only (open/close/populate/read) — API calls and chat rendering
   are orchestrated by App, same split as CropPicker/DiseaseCheck. ───── */
const CropAdvisorModal = (() => {
  let _overlay, _talukaSelect, _monthSelect, _soilTypeSelect, _prevCropSelect;
  let _irrigationGroup, _submitBtn;
  let _onSubmit = null;
  let _options = null; // cached {talukas, soil_types, crops} from API.cropOptions()
  let _irrigationValue = "yes";

  function _localized(entry, lang) {
    return (entry && (entry[lang] || entry.en)) || "";
  }

  function _populateSelect(select, items, { valueKey, labelFn, includeNone = false, noneLabel = "" }) {
    const previous = select.value;
    select.innerHTML = "";
    if (includeNone) {
      const opt = document.createElement("option");
      opt.value = "";
      opt.textContent = noneLabel;
      select.appendChild(opt);
    }
    items.forEach((item) => {
      const opt = document.createElement("option");
      opt.value = valueKey(item);
      opt.textContent = labelFn(item);
      select.appendChild(opt);
    });
    if (previous) select.value = previous;
  }

  function _populateMonths(lang) {
    const previous = _monthSelect.value;
    _monthSelect.innerHTML = "";
    for (let m = 1; m <= 12; m++) {
      const opt = document.createElement("option");
      opt.value = String(m);
      opt.textContent = CropI18n.monthName(m, lang);
      _monthSelect.appendChild(opt);
    }
    // Default to the current month the first time; keep the farmer's
    // choice on later language switches.
    _monthSelect.value = previous || String(new Date().getMonth() + 1);
  }

  function _populateOptions(options, lang) {
    const labels = CropI18n.getLabels(lang);
    _populateSelect(_talukaSelect, options.talukas || [], {
      valueKey: (t) => t.en, // crop_api matches talukas by English name
      labelFn: (t) => _localized(t, lang),
    });
    _populateSelect(_soilTypeSelect, options.soil_types || [], {
      valueKey: (s) => s.id,
      labelFn: (s) => _localized(s, lang),
      includeNone: true,
      noneLabel: labels.notSure,
    });
    _populateSelect(_prevCropSelect, options.crops || [], {
      valueKey: (c) => c.id,
      labelFn: (c) => _localized(c.names, lang),
      includeNone: true,
      noneLabel: labels.none,
    });
    _populateMonths(lang);
  }

  function _applyIrrigationSelection() {
    _irrigationGroup.querySelectorAll(".toggle-btn").forEach((btn) => {
      btn.classList.toggle("selected", btn.dataset.value === _irrigationValue);
    });
  }

  function applyLanguage() {
    const lang = LanguageSelector.getLanguage();
    const labels = CropI18n.getLabels(lang);
    document.getElementById("crop-advisor-modal-title").textContent = labels.modalTitle;
    document.getElementById("crop-advisor-taluka-label").textContent = labels.taluka;
    document.getElementById("crop-advisor-month-label").textContent = labels.sowingMonth;
    document.getElementById("crop-advisor-irrigation-label").textContent = labels.irrigation;
    document.getElementById("crop-advisor-soiltype-label").textContent = labels.soilType;
    document.getElementById("crop-advisor-prevcrop-label").textContent = labels.previousCrop;
    document.getElementById("crop-advisor-soilcard-label").textContent = labels.soilCard;
    document.getElementById("crop-advisor-ph-label").textContent = labels.ph;
    document.getElementById("crop-advisor-ec-label").textContent = labels.ec;
    document.getElementById("crop-advisor-oc-label").textContent = labels.oc;
    document.getElementById("crop-advisor-n-label").textContent = labels.n;
    document.getElementById("crop-advisor-p-label").textContent = labels.p;
    document.getElementById("crop-advisor-k-label").textContent = labels.k;
    _submitBtn.textContent = labels.submit;
    _irrigationGroup.querySelector('[data-value="yes"]').textContent = labels.yes;
    _irrigationGroup.querySelector('[data-value="no"]').textContent = labels.no;

    if (_options) _populateOptions(_options, lang);
  }

  function init(onSubmit) {
    _onSubmit = onSubmit;
    _overlay = document.getElementById("crop-advisor-overlay");
    _talukaSelect = document.getElementById("crop-advisor-taluka");
    _monthSelect = document.getElementById("crop-advisor-month");
    _soilTypeSelect = document.getElementById("crop-advisor-soiltype");
    _prevCropSelect = document.getElementById("crop-advisor-prevcrop");
    _irrigationGroup = document.getElementById("crop-advisor-irrigation");
    _submitBtn = document.getElementById("crop-advisor-submit");

    document.getElementById("crop-advisor-close").addEventListener("click", close);
    _overlay.addEventListener("click", (e) => {
      if (e.target === _overlay) close();
    });

    _irrigationGroup.querySelectorAll(".toggle-btn").forEach((btn) => {
      btn.addEventListener("click", () => {
        _irrigationValue = btn.dataset.value;
        _applyIrrigationSelection();
      });
    });
    _applyIrrigationSelection();

    _submitBtn.addEventListener("click", () => {
      if (_onSubmit) _onSubmit(_readForm());
    });

    applyLanguage();
  }

  function open(options) {
    _options = options;
    _irrigationValue = "yes";
    _applyIrrigationSelection();
    applyLanguage();
    _overlay.classList.add("open");
  }

  function close() {
    _overlay.classList.remove("open");
  }

  function setSubmitEnabled(enabled) {
    _submitBtn.disabled = !enabled;
  }

  function _numberOrUndefined(id) {
    const val = document.getElementById(id).value.trim();
    return val === "" ? undefined : Number(val);
  }

  function _readForm() {
    return {
      taluka: _talukaSelect.value,
      sowing_month: Number(_monthSelect.value),
      irrigation: _irrigationValue,
      previous_crop: _prevCropSelect.value || null,
      soil_type: _soilTypeSelect.value || null,
      soil: {
        ph: _numberOrUndefined("crop-advisor-ph"),
        ec: _numberOrUndefined("crop-advisor-ec"),
        oc: _numberOrUndefined("crop-advisor-oc"),
        n: _numberOrUndefined("crop-advisor-n"),
        p: _numberOrUndefined("crop-advisor-p"),
        k: _numberOrUndefined("crop-advisor-k"),
      },
    };
  }

  return { init, open, close, applyLanguage, setSubmitEnabled };
})();


/* ─── App ─────────────────────────────────────────────────── */
const App = (() => {
  let _micBtn, _micLabel;
  let _textInput, _sendBtn;
  let _transcriptEl;
  let _photoBtn, _photoInput;
  let _cropAdvisorBtn, _cropAdvisorLabel;
  let _pendingCrop = null;
  let _isProcessing = false;
  let _interimText = "";
  let _finalLiveText = "";
  let _cropOptionsCache = null;

  function init() {
    // DOM refs
    _micBtn = document.getElementById("mic-btn");
    _micLabel = document.getElementById("mic-label");
    _textInput = document.getElementById("text-input");
    _sendBtn = document.getElementById("send-btn");
    _transcriptEl = document.getElementById("input-transcript");
    _photoBtn = document.getElementById("photo-btn");
    _photoInput = document.getElementById("photo-input");
    _cropAdvisorBtn = document.getElementById("crop-advisor-btn");
    _cropAdvisorLabel = document.getElementById("crop-advisor-label");

    const chatArea = document.getElementById("chat-area");
    const welcomeScreen = document.getElementById("welcome-screen");

    ChatUI.init(chatArea, welcomeScreen);
    LanguageSelector.init();
    Settings.init();
    ChatHistory.init();
    CropPicker.init((crop) => {
      _pendingCrop = crop;
      _photoInput.click();
    });
    CropAdvisorModal.init((formData) => _handleCropAdvisorSubmit(formData));

    // Re-localize the Crop Advisor button + modal whenever the farmer
    // switches language (the rest of the app's UI text doesn't localize
    // today, but the Crop Advisor's labels are built to, from the start).
    LanguageSelector.onChange(() => {
      _updateCropAdvisorButtonLabel();
      CropAdvisorModal.applyLanguage();
    });
    _updateCropAdvisorButtonLabel();

    // Settings button
    document.getElementById("settings-btn").addEventListener("click", Settings.open);

    // Mic button
    _micBtn.addEventListener("click", _toggleMic);

    // Photo button (leaf disease check)
    _photoBtn.addEventListener("click", CropPicker.open);
    _photoInput.addEventListener("change", () => {
      const file = _photoInput.files && _photoInput.files[0];
      const crop = _pendingCrop;
      _photoInput.value = ""; // allow picking the same file again next time
      if (file && crop) {
        _handlePhotoUpload(file, crop);
      }
    });

    // Crop Advisor button
    _cropAdvisorBtn.addEventListener("click", _openCropAdvisor);

    // Text input
    _textInput.addEventListener("keydown", (e) => {
      if (e.key === "Enter" && !e.shiftKey) {
        e.preventDefault();
        _sendText();
      }
    });
    _textInput.addEventListener("input", _autoResize);
    _sendBtn.addEventListener("click", _sendText);

    // Check backend health
    _checkHealth();

    // PWA install prompt
    _setupInstallPrompt();

    // Register service worker
    _registerSW();

    // "No internet" notice
    _setupOfflineNotice();
  }

  // ── Mic: toggle recording ────────────────────────────────
  let _micStarting = false;

  async function _toggleMic() {
    if (_isProcessing || _micStarting) return;

    if (SpeechInput.isRecording()) {
      // Stop recording
      SpeechInput.stopRecording();
      _micBtn.classList.remove("recording");
      _micBtn.classList.add("processing");
      _setMicLabel("Processing...");
      _isProcessing = true;
      return;
    }

    // Start recording
    _micStarting = true;
    _micBtn.classList.add("recording");
    _setMicLabel("Tap to stop...");
    _interimText = "";
    _finalLiveText = "";
    _showTranscript("");

    const stream = await SpeechInput.startRecording({
      onInterim: (text, finalPart, interimPart) => {
        _finalLiveText = finalPart;
        _interimText = interimPart;
        _showTranscript(finalPart, interimPart);
      },
      onFinal: (audioBlob) => {
        _processSpeech(audioBlob);
      },
      onError: (msg) => {
        _micBtn.classList.remove("recording");
        _setMicLabel("Tap to speak");
        _hideTranscript();
        Toast.show(msg, "error");
        _isProcessing = false;
        _micStarting = false;
      },
    });

    if (!stream) {
      _micBtn.classList.remove("recording");
      _setMicLabel("Tap to speak");
    }

    _micStarting = false;
  }

  // ── Process speech audio blob → API ────────────────────
  async function _processSpeech(audioBlob) {
    if (audioBlob.size < 1000) {
      // Too small — probably no speech
      _micBtn.classList.remove("processing");
      _setMicLabel("Tap to speak");
      _hideTranscript();
      _isProcessing = false;
      Toast.show("No speech detected. Try again!", "info");
      return;
    }

    const typingEl = ChatUI.addTypingBubble();

    try {
      const result = await API.speechToText(audioBlob);
      typingEl.remove();
      _hideTranscript();

      if (result.marathi_text) {
        ChatUI.addSpeechResult(result.marathi_text, result.english_text);

        // Start the LLM call right away; saving history runs alongside it
        const aiTypingEl = ChatUI.addTypingBubble();
        const chatPromise = API.chat(result.marathi_text);
        const historyReady = ChatHistory.ensureChatId(result.marathi_text.substring(0, 30) || "Speech Chat")
          .then(() => API.saveMessage(ChatHistory.getChatId(), "user", "speech_result", {
            marathi_text: result.marathi_text,
            english_text: result.english_text
          }));
        try {
          const chatResult = await chatPromise;
          const aiResponseText = chatResult.response;
          aiTypingEl.remove();

          const ttsLanguage = chatResult.translation_failed ? "en" : LanguageSelector.getLanguage();
          let ttsResult = null;
          try {
            ttsResult = await API.textToSpeech(aiResponseText, Config.get("voice"), ttsLanguage);
          } catch (ttsErr) {
            console.warn("TTS failed, showing text only:", ttsErr);
          }

          if (ttsResult) {
            ChatUI.addTTSResult(
              ttsResult.marathi_text,
              ttsResult.word_timings,
              ttsResult.audio_base64,
              ttsResult.voice || Config.get("voice"),
              ttsResult.duration || 3,
            );
          } else {
            ChatUI.addBotText("Voice not available", aiResponseText);
          }

          await historyReady;
          if (ttsResult) {
            await API.saveMessage(ChatHistory.getChatId(), "bot", "bot_tts", {
              ai_response: aiResponseText,
              reply_language: LanguageSelector.getLanguage(),
              marathi_text: ttsResult.marathi_text,
              word_timings: ttsResult.word_timings,
              audio_base64: ttsResult.audio_base64,
              voice: ttsResult.voice || Config.get("voice"),
              duration: ttsResult.duration || 3
            });
          } else {
            await API.saveMessage(ChatHistory.getChatId(), "bot", "bot_text", {
              ai_response: aiResponseText,
              reply_language: LanguageSelector.getLanguage(),
            });
          }
        } catch (err) {
          aiTypingEl.remove();
          ChatUI.addError("AI Error: " + err.message);
        }
      } else {
        ChatUI.addError("No speech detected. Please try again.");
      }
    } catch (err) {
      typingEl.remove();
      _hideTranscript();
      ChatUI.addError(err.message || "Speech recognition failed.");
      Toast.show("Error: " + err.message, "error");
    } finally {
      _micBtn.classList.remove("processing");
      _setMicLabel("Tap to speak");
      _isProcessing = false;
    }
  }

  // ── Leaf photo upload → disease check ───────────────────
  async function _handlePhotoUpload(file, crop) {
    if (_isProcessing) return;
    _isProcessing = true;

    const thumbnailUrl = URL.createObjectURL(file);
    ChatUI.addUserPhoto(thumbnailUrl, crop);
    const loadingEl = ChatUI.addDiseaseLoading("Checking your leaf...");

    try {
      const result = await DiseaseCheck.checkDisease(file, crop);
      loadingEl.remove();
      ChatUI.addDiseaseResult(result);
      await _speakDiseaseMessage(result);
    } catch (err) {
      loadingEl.remove();
      ChatUI.addError(_friendlyDiseaseError(err));
    } finally {
      _isProcessing = false;
    }
  }

  // ── Speak only the result message (no percentages) ─────
  async function _speakDiseaseMessage(result) {
    try {
      const ttsResult = await API.textToSpeech(result.message, Config.get("voice"));

      ChatUI.addTTSResult(
        ttsResult.marathi_text,
        ttsResult.word_timings,
        ttsResult.audio_base64,
        ttsResult.voice || Config.get("voice"),
        ttsResult.duration || 3,
      );
    } catch (err) {
      console.warn("Could not speak disease result:", err);
    }
  }

  // ── Turn a disease-check error into a farmer-friendly message ──
  function _friendlyDiseaseError(err) {
    const status = err.status;
    const raw = err.message || "";

    if (status === 413) {
      return "That photo is too large. Please use a smaller photo (under 10 MB) and try again.";
    }
    if (status === 400 && /readable image|JPG or PNG/i.test(raw)) {
      return "That file doesn't look like a photo. Please upload a JPG or PNG image.";
    }
    if (status === 503) {
      return "The leaf-check service isn't reachable right now. Please make sure it's running and try again.";
    }
    return raw || "Something went wrong while checking the leaf. Please try again.";
  }

  // ── Crop Advisor: button + modal → /api/crop ────────────
  function _updateCropAdvisorButtonLabel() {
    const label = CropI18n.t("buttonLabel", LanguageSelector.getLanguage());
    _cropAdvisorBtn.title = label;
    _cropAdvisorBtn.setAttribute("aria-label", label);
    _cropAdvisorLabel.textContent = label;
  }

  async function _openCropAdvisor() {
    if (_isProcessing) return;
    const lang = LanguageSelector.getLanguage();
    try {
      if (!_cropOptionsCache) {
        _cropOptionsCache = await API.cropOptions();
      }
      CropAdvisorModal.open(_cropOptionsCache);
    } catch (err) {
      console.warn("Could not load crop advisor options:", err);
      ChatUI.addError(CropI18n.t("optionsError", lang));
    }
  }

  async function _handleCropAdvisorSubmit(formData) {
    if (_isProcessing) return;
    _isProcessing = true;
    CropAdvisorModal.setSubmitEnabled(false);
    CropAdvisorModal.close();

    const lang = LanguageSelector.getLanguage();
    const labels = CropI18n.getLabels(lang);
    const loadingEl = ChatUI.addDiseaseLoading(labels.loading);

    try {
      const result = await API.cropAdvice({ ...formData, language: lang });
      loadingEl.remove();
      if (Array.isArray(result.crops) && result.crops.length > 0) {
        ChatUI.addCropResult(result, labels, (speakBtn) => _speakCropAdvice(result, lang, speakBtn));
      } else {
        // service_down/no_crops/400 all land here as a normal message —
        // result.message (or disclaimer alone) is already in the farmer's language.
        const text = [result.message, result.disclaimer].filter(Boolean).join("\n\n");
        ChatUI.addBotText(labels.resultLabel, text || labels.timeoutError);
      }
    } catch (err) {
      loadingEl.remove();
      ChatUI.addError(_friendlyCropError(err, lang));
    } finally {
      _isProcessing = false;
      CropAdvisorModal.setSubmitEnabled(true);
    }
  }

  // ── Speak only names/reason/risk/disclaimer — not source/evidence —
  //    and only when the farmer taps 🔊 (this text is long, unlike the
  //    short disease message, so it is never spoken automatically) ────
  function _buildCropSpokenText(result) {
    const parts = [];
    (result.crops || []).forEach((crop) => {
      if (crop.name) parts.push(crop.name);
      if (crop.reason) parts.push(crop.reason);
      if (crop.risk) parts.push(crop.risk);
    });
    if (result.disclaimer) parts.push(result.disclaimer);
    return parts.join(". ");
  }

  async function _speakCropAdvice(result, language, speakBtn) {
    const text = _buildCropSpokenText(result);
    if (!text) return;

    speakBtn.disabled = true;
    speakBtn.classList.add("playing");
    try {
      const ttsResult = await API.textToSpeech(text, Config.get("voice"), language);
      ChatUI.addTTSResult(
        ttsResult.marathi_text,
        ttsResult.word_timings,
        ttsResult.audio_base64,
        ttsResult.voice || Config.get("voice"),
        ttsResult.duration || 3,
      );
    } catch (err) {
      // Voice failed — the card's text is already showing, so just log it.
      console.warn("Could not speak crop advice:", err);
    } finally {
      speakBtn.disabled = false;
      speakBtn.classList.remove("playing");
    }
  }

  // ── Turn a /api/crop error into a farmer-friendly message ──────────
  function _friendlyCropError(err, lang) {
    if (err && (err.name === "TimeoutError" || err.name === "AbortError")) {
      return CropI18n.t("timeoutError", lang);
    }
    return (err && err.message) || CropI18n.t("timeoutError", lang);
  }

  // ── Send English text → API ─────────────────────────────
  async function _sendText() {
    const text = _textInput.value.trim();
    if (!text || _isProcessing) return;

    _isProcessing = true;
    _textInput.value = "";
    _autoResize();
    _sendBtn.disabled = true;

    ChatUI.addUserText(text);
    const typingEl = ChatUI.addTypingBubble();

    try {
      // Start the LLM call right away; saving history runs alongside it
      const chatPromise = API.chat(text);
      const historyReady = ChatHistory.ensureChatId(text.substring(0, 30))
        .then(() => API.saveMessage(ChatHistory.getChatId(), "user", "user_text", {
          english_text: text
        }));

      const chatResult = await chatPromise;
      const aiResponseText = chatResult.response;
      typingEl.remove();

      const ttsLanguage = chatResult.translation_failed ? "en" : LanguageSelector.getLanguage();
      let result = null;
      try {
        result = await API.textToSpeech(aiResponseText, Config.get("voice"), ttsLanguage);
      } catch (ttsErr) {
        console.warn("TTS failed, showing text only:", ttsErr);
      }

      if (result) {
        ChatUI.addTTSResult(
          result.marathi_text,
          result.word_timings,
          result.audio_base64,
          result.voice || Config.get("voice"),
          result.duration || 3,
        );
      } else {
        ChatUI.addBotText("Voice not available", aiResponseText);
      }

      await historyReady;
      if (result) {
        await API.saveMessage(ChatHistory.getChatId(), "bot", "bot_tts", {
          ai_response: aiResponseText,
          reply_language: LanguageSelector.getLanguage(),
          marathi_text: result.marathi_text,
          word_timings: result.word_timings,
          audio_base64: result.audio_base64,
          voice: result.voice || Config.get("voice"),
          duration: result.duration || 3
        });
      } else {
        await API.saveMessage(ChatHistory.getChatId(), "bot", "bot_text", {
          ai_response: aiResponseText,
          reply_language: LanguageSelector.getLanguage(),
        });
      }
    } catch (err) {
      typingEl.remove();
      ChatUI.addError(err.message || "Request failed.");
      Toast.show("Error: " + err.message, "error");
    } finally {
      _isProcessing = false;
      _sendBtn.disabled = false;
    }
  }

  // ── UI helpers ──────────────────────────────────────────
  function _setMicLabel(text) {
    if (_micLabel) _micLabel.textContent = text;
  }

  function _showTranscript(finalText, interimText = "") {
    _transcriptEl.classList.add("visible");
    _transcriptEl.innerHTML = "";
    if (finalText) {
      const finalSpan = document.createElement("span");
      finalSpan.textContent = finalText;
      _transcriptEl.appendChild(finalSpan);
    }
    if (interimText) {
      const interim = document.createElement("span");
      interim.className = "interim";
      interim.textContent = (finalText ? " " : "") + interimText;
      _transcriptEl.appendChild(interim);
    }
    if (!finalText && !interimText) {
      const ph = document.createElement("span");
      ph.className = "live-placeholder";
      ph.textContent = "बोलत आहे... (Listening...)";
      _transcriptEl.appendChild(ph);
    }
  }

  function _hideTranscript() {
    _transcriptEl.classList.remove("visible");
    _transcriptEl.innerHTML = "";
  }

  function _autoResize() {
    _textInput.style.height = "auto";
    _textInput.style.height = Math.min(_textInput.scrollHeight, 140) + "px";
  }

  // ── Backend health check ────────────────────────────────
  async function _checkHealth() {
    const indicator = document.getElementById("status-indicator");
    const statusText = document.getElementById("status-text");

    try {
      const health = await API.healthCheck();
      indicator.classList.add("connected");
      indicator.classList.remove("error");
      statusText.textContent = `Connected · ${health.inference_mode}`;
      // On phones only the dot is visible; the title shows the text on long-press/hover
      indicator.title = statusText.textContent;
    } catch {
      indicator.classList.remove("connected");
      indicator.classList.add("error");
      statusText.textContent = "Backend offline";
      indicator.title = statusText.textContent;
      Toast.show(
        "⚠️ Backend not reachable. Start the server at " + Config.get("apiUrl"),
        "error",
        6000
      );
    }
  }

  // ── PWA install prompt ──────────────────────────────────
  let _deferredPrompt = null;

  function _setupInstallPrompt() {
    window.addEventListener("beforeinstallprompt", (e) => {
      e.preventDefault();
      _deferredPrompt = e;

      // Banner text in the farmer's chosen language
      const INSTALL_TEXT = {
        mr: { title: "अ‍ॅप इन्स्टॉल करा", body: "AgriSaathi होम स्क्रीनवर जोडा", btn: "इन्स्टॉल" },
        hi: { title: "ऐप इंस्टॉल करें", body: "AgriSaathi को होम स्क्रीन पर जोड़ें", btn: "इंस्टॉल" },
        en: { title: "Install App", body: "Add AgriSaathi to your home screen", btn: "Install" },
      };
      const t = INSTALL_TEXT[LanguageSelector.getLanguage()] || INSTALL_TEXT.en;

      const prompt = document.createElement("div");
      prompt.className = "install-prompt";
      prompt.innerHTML = `
        <div class="install-text">
          <strong>📱 ${t.title}</strong><br>
          ${t.body}
        </div>
        <button class="install-btn" id="install-btn">${t.btn}</button>
        <button class="install-dismiss" id="install-dismiss" aria-label="Close">✕</button>
      `;
      document.body.appendChild(prompt);

      document.getElementById("install-btn").addEventListener("click", async () => {
        _deferredPrompt.prompt();
        const { outcome } = await _deferredPrompt.userChoice;
        if (outcome === "accepted") {
          Toast.show("App installed! 🎉", "success");
        }
        prompt.remove();
        _deferredPrompt = null;
      });

      document.getElementById("install-dismiss").addEventListener("click", () => {
        prompt.remove();
      });
    });
  }

  // ── Service Worker registration + "new version" banner ──
  // When sw.js changes (new CACHE_NAME), the phone downloads the new
  // version in the background. It waits until the farmer taps
  // "Refresh", then the page reloads once with the new version.
  let _userAskedUpdate = false;

  function _registerSW() {
    if (!("serviceWorker" in navigator)) return;

    window.addEventListener("load", () => {
      navigator.serviceWorker.register("./sw.js").then((reg) => {
        // A new version finished downloading on an earlier visit
        if (reg.waiting && navigator.serviceWorker.controller) {
          _showUpdateBanner(reg.waiting);
        }
        // A new version starts downloading now
        reg.addEventListener("updatefound", () => {
          const newWorker = reg.installing;
          if (!newWorker) return;
          newWorker.addEventListener("statechange", () => {
            // "controller" is null on the very first install: nothing to update then
            if (newWorker.state === "installed" && navigator.serviceWorker.controller) {
              _showUpdateBanner(newWorker);
            }
          });
        });
        // If the app stays open for long, check for a new version every hour
        setInterval(() => reg.update().catch(() => {}), 60 * 60 * 1000);
      }).catch((err) => {
        console.warn("SW registration failed:", err);
      });
    });

    // The new version took over: reload once to use its files.
    // Only after the farmer tapped Refresh (not on the first install).
    navigator.serviceWorker.addEventListener("controllerchange", () => {
      if (!_userAskedUpdate) return;
      _userAskedUpdate = false;
      window.location.reload();
    });
  }

  function _showUpdateBanner(waitingWorker) {
    if (document.getElementById("update-banner")) return;

    const UPDATE_TEXT = {
      mr: { title: "नवीन आवृत्ती उपलब्ध", body: "नवीन सुविधा वापरण्यासाठी रिफ्रेश करा", btn: "रिफ्रेश" },
      hi: { title: "नया संस्करण उपलब्ध", body: "नई सुविधाओं के लिए रिफ्रेश करें", btn: "रिफ्रेश" },
      en: { title: "New version available", body: "Refresh to get the latest features", btn: "Refresh" },
    };
    const t = UPDATE_TEXT[LanguageSelector.getLanguage()] || UPDATE_TEXT.en;

    // Same look and position as the install banner
    const banner = document.createElement("div");
    banner.className = "install-prompt";
    banner.id = "update-banner";
    banner.setAttribute("role", "status");
    banner.innerHTML = `
      <div class="install-text">
        <strong>🔄 ${t.title}</strong><br>
        ${t.body}
      </div>
      <button class="install-btn" id="update-btn">${t.btn}</button>
      <button class="install-dismiss" id="update-dismiss" aria-label="Close">✕</button>
    `;
    document.body.appendChild(banner);

    document.getElementById("update-btn").addEventListener("click", () => {
      _userAskedUpdate = true;
      waitingWorker.postMessage("SKIP_WAITING");
      banner.remove();
    });
    // "Later": the new version is used next time the app is fully closed and opened
    document.getElementById("update-dismiss").addEventListener("click", () => {
      banner.remove();
    });
  }

  // ── Offline notice ──────────────────────────────────────
  // The app opens offline (cached), but answers come from the
  // backend, so tell the farmer clearly instead of showing errors.
  function _setupOfflineNotice() {
    const OFFLINE_TEXT = {
      mr: "इंटरनेट नाही. प्रश्न, फोटो तपासणी आणि पीक सल्ल्यासाठी इंटरनेट लागते.",
      hi: "इंटरनेट नहीं है। सवाल, फोटो जांच और फसल सलाह के लिए इंटरनेट चाहिए।",
      en: "No internet. Chat, photo check and crop advice need a connection.",
    };

    const notice = document.createElement("div");
    notice.className = "offline-notice";
    notice.id = "offline-notice";
    notice.setAttribute("role", "status");
    notice.setAttribute("aria-live", "polite");
    // First thing inside #app, above the header, so it pushes the page down
    const app = document.getElementById("app");
    if (!app) return;
    app.insertBefore(notice, app.firstChild);

    function update() {
      const lang = LanguageSelector.getLanguage();
      notice.textContent = "📵 " + (OFFLINE_TEXT[lang] || OFFLINE_TEXT.en);
      notice.hidden = navigator.onLine;
    }

    window.addEventListener("offline", update);
    window.addEventListener("online", () => {
      update();
      _checkHealth(); // turn the status dot green again
    });
    LanguageSelector.onChange(update);
    update();
  }

  return { init };
})();



/* ─── Router Module (Multi-page view switching & hash routing) ─── */
const Router = (() => {
  const ROUTES = ["dashboard", "crop", "disease", "chat"];
  let _currentRoute = "dashboard";

  function normalizeRoute(hash) {
    const clean = (hash || "").replace(/^#\/?/, "").toLowerCase().trim();
    return ROUTES.includes(clean) ? clean : "dashboard";
  }

  function getCurrentRoute() {
    return _currentRoute;
  }

  function navigate(route) {
    const target = normalizeRoute(route);
    if (!Auth.isLoggedIn()) {
      // If not authenticated, ensure login screen is presented
      const authScreen = document.getElementById("auth-screen");
      const layoutContainer = document.getElementById("layout-container");
      if (authScreen) authScreen.style.display = "flex";
      if (layoutContainer) layoutContainer.style.display = "none";
      return;
    }

    _currentRoute = target;
    window.location.hash = `#/${target}`;

    // Update views visibility
    ROUTES.forEach((r) => {
      const viewEl = document.getElementById(`view-${r}`);
      if (viewEl) {
        if (r === target) {
          viewEl.classList.remove("hidden");
        } else {
          viewEl.classList.add("hidden");
        }
      }
    });

    // Update Desktop Nav links
    document.querySelectorAll(".desktop-nav-link").forEach((link) => {
      link.classList.toggle("active", link.dataset.route === target);
    });

    // Update Sidebar Nav items
    document.querySelectorAll(".sidebar-nav-item").forEach((item) => {
      item.classList.toggle("active", item.dataset.route === target);
    });

    // Update Mobile Bottom Nav items
    document.querySelectorAll(".bottom-nav-item").forEach((item) => {
      item.classList.toggle("active", item.dataset.route === target);
    });

    // Close mobile sidebar if open
    const sidebar = document.getElementById("history-sidebar");
    if (sidebar && sidebar.classList.contains("open")) {
      sidebar.classList.remove("open");
    }

    // Scroll view to top (except chat which might want bottom scroll)
    const viewsContainer = document.getElementById("views-container");
    if (viewsContainer && target !== "chat") {
      viewsContainer.scrollTop = 0;
    }

    // Route-specific activation hooks
    if (target === "crop") {
      CropPage.onEnter();
    } else if (target === "disease") {
      DiseasePage.onEnter();
    } else if (target === "chat") {
      const chatArea = document.getElementById("chat-area");
      if (chatArea) chatArea.scrollTop = chatArea.scrollHeight;
    }
  }

  function init() {
    // Listen for hash changes (e.g. browser back/forward buttons)
    window.addEventListener("hashchange", () => {
      navigate(window.location.hash);
    });

    // Delegated click handler for all elements with data-route
    document.addEventListener("click", (e) => {
      const routeEl = e.target.closest("[data-route]");
      if (routeEl) {
        e.preventDefault();
        const route = routeEl.getAttribute("data-route");
        if (route) navigate(route);
      }
    });
  }

  return { init, navigate, getCurrentRoute };
})();


/* ─── CropPage (Dedicated Crop Recommendation View) ───────── */
const CropPage = (() => {
  let _talukaSelect, _monthSelect, _soilTypeSelect, _prevCropSelect;
  let _irrigationValue = "yes";
  let _cachedOptions = null;
  let _lastResult = null;

  const NASHIK_TALUKAS = [
    { en: "Nashik", mr: "नाशिक", hi: "नासिक" },
    { en: "Niphad", mr: "निफाड", hi: "निफाड" },
    { en: "Dindori", mr: "दिंडोरी", hi: "दिंडोरी" },
    { en: "Sinnar", mr: "सिन्नर", hi: "सिन्नर" },
    { en: "Yeola", mr: "येवला", hi: "येवला" },
    { en: "Chandwad", mr: "चांदवड", hi: "चांदवड" },
    { en: "Malegaon", mr: "मालेगाव", hi: "मालेगांव" },
    { en: "Baglan", mr: "बागलाण (सटाणा)", hi: "बागलाण" },
    { en: "Kalwan", mr: "कळवण", hi: "कलवण" },
    { en: "Deola", mr: "देवळा", hi: "देवला" },
    { en: "Igatpuri", mr: "इगतपुरी", hi: "इगतपुरी" },
    { en: "Trimbakeshwar", mr: "त्र्यंबकेश्वर", hi: "त्र्यंबकेश्वर" },
    { en: "Peint", mr: "पेठ", hi: "पेठ" },
    { en: "Surgana", mr: "सुरगाणा", hi: "सुरगाणा" },
    { en: "Nandgaon", mr: "नांदगाव", hi: "नांदगांव" },
  ];

  function populateSelect(select, items, valueFn, textFn, defaultVal = "") {
    if (!select) return;
    const current = select.value || defaultVal;
    select.innerHTML = "";
    items.forEach((item) => {
      const opt = document.createElement("option");
      opt.value = valueFn(item);
      opt.textContent = textFn(item);
      select.appendChild(opt);
    });
    if (current) select.value = current;
  }

  async function loadOptions() {
    const lang = LanguageSelector.getLanguage();
    try {
      if (!_cachedOptions) {
        _cachedOptions = await API.cropOptions();
      }
      applyOptions(_cachedOptions, lang);
    } catch {
      applyOptions({ talukas: NASHIK_TALUKAS, soil_types: [], crops: [] }, lang);
    }
  }

  function applyOptions(options, lang) {
    const talukas = (options.talukas && options.talukas.length > 0) ? options.talukas : NASHIK_TALUKAS;
    populateSelect(
      _talukaSelect,
      talukas,
      (t) => t.en,
      (t) => (t[lang] || t.en)
    );

    // Populate months (1-12)
    const monthItems = [];
    for (let m = 1; m <= 12; m++) {
      monthItems.push({ val: m, name: CropI18n.monthName(m, lang) });
    }
    populateSelect(
      _monthSelect,
      monthItems,
      (m) => String(m.val),
      (m) => m.name,
      String(new Date().getMonth() + 1)
    );

    // Populate previous crops if available
    if (options.crops && options.crops.length > 0 && _prevCropSelect) {
      _prevCropSelect.innerHTML = `<option value="">${CropI18n.getLabels(lang).none || "None / काही नाही"}</option>`;
      options.crops.forEach((c) => {
        const opt = document.createElement("option");
        opt.value = c.id;
        opt.textContent = (c.names && c.names[lang]) || (c.names && c.names.en) || c.id;
        _prevCropSelect.appendChild(opt);
      });
    }
  }

  function onEnter() {
    loadOptions();
  }

  function getNumberOrUndefined(id) {
    const val = document.getElementById(id)?.value?.trim();
    if (val === "" || val === undefined || isNaN(val)) return undefined;
    return Number(val);
  }

  function showAlert(msg) {
    const alertEl = document.getElementById("crop-form-alert");
    if (alertEl) {
      alertEl.textContent = msg;
      alertEl.className = "form-alert error";
      alertEl.classList.remove("hidden");
    }
  }

  function clearAlert() {
    const alertEl = document.getElementById("crop-form-alert");
    if (alertEl) alertEl.classList.add("hidden");
  }

  async function handleSubmit(e) {
    e.preventDefault();
    clearAlert();

    const taluka = _talukaSelect?.value?.trim();
    const month = Number(_monthSelect?.value);
    const n = getNumberOrUndefined("crop-n");
    const p = getNumberOrUndefined("crop-p");
    const k = getNumberOrUndefined("crop-k");
    const ph = getNumberOrUndefined("crop-ph");
    const temp = getNumberOrUndefined("crop-temperature");
    const humidity = getNumberOrUndefined("crop-humidity");
    const rainfall = getNumberOrUndefined("crop-rainfall");
    const soilType = _soilTypeSelect?.value || null;
    const prevCrop = _prevCropSelect?.value || null;
    const lang = LanguageSelector.getLanguage();

    if (!taluka) {
      showAlert(lang === "mr" ? "कृपया तालुका निवडा." : lang === "hi" ? "कृपया तालुका चुनें।" : "Please select a taluka.");
      return;
    }
    if (!month || month < 1 || month > 12) {
      showAlert(lang === "mr" ? "कृपया पेरणीचा महिना निवडा." : lang === "hi" ? "कृपया बुआई का महीना चुनें।" : "Please select sowing month.");
      return;
    }
    if (ph !== undefined && (ph < 0 || ph > 14)) {
      showAlert(lang === "mr" ? "मातीचा pH ० ते १४ च्या दरम्यान असावा." : lang === "hi" ? "मिट्टी का pH 0 से 14 के बीच होना चाहिए।" : "Soil pH must be between 0 and 14.");
      return;
    }

    const payload = {
      taluka: taluka,
      sowing_month: month,
      irrigation: _irrigationValue,
      soil_type: soilType,
      previous_crop: prevCrop,
      soil: {
        ph: ph,
        n: n,
        p: p,
        k: k,
        temperature: temp,
        humidity: humidity,
        rainfall: rainfall
      },
      language: lang
    };

    const loadingEl = document.getElementById("crop-loading-state");
    const errorBox = document.getElementById("crop-error-box");
    const resultsContainer = document.getElementById("crop-results-container");
    const submitBtn = document.getElementById("btn-submit-crop");

    if (errorBox) errorBox.classList.add("hidden");
    if (resultsContainer) resultsContainer.classList.add("hidden");
    if (loadingEl) loadingEl.classList.remove("hidden");
    if (submitBtn) submitBtn.disabled = true;

    try {
      const result = await API.cropAdvice(payload);
      _lastResult = result;
      if (loadingEl) loadingEl.classList.add("hidden");
      renderResults(result, lang);
    } catch (err) {
      if (loadingEl) loadingEl.classList.add("hidden");
      if (errorBox) {
        const errorMsgEl = document.getElementById("crop-error-message");
        if (errorMsgEl) {
          errorMsgEl.textContent = err.message || (lang === "mr" ? "पीक शिफारस सर्व्हर उपलब्ध नाही." : "Crop recommendation service is offline.");
        }
        errorBox.classList.remove("hidden");
        errorBox.scrollIntoView({ behavior: "smooth", block: "nearest" });
      }
    } finally {
      if (submitBtn) submitBtn.disabled = false;
    }
  }

  function renderResults(result, lang) {
    const container = document.getElementById("crop-results-container");
    const listEl = document.getElementById("crop-results-list");
    const disclaimerEl = document.getElementById("crop-results-disclaimer");

    if (!container || !listEl) return;
    listEl.innerHTML = "";

    const crops = result.crops || [];

    if (crops.length === 0) {
      listEl.innerHTML = `
        <div class="crop-result-card" style="border-left-color: var(--c-ochre);">
          <div class="crop-card-top">
            <h3 class="crop-card-name">${lang === "mr" ? "कोणतीही थेट पिके जुळली नाहीत" : "No Direct Crop Match"}</h3>
          </div>
          <div class="crop-card-reason">
            ${result.message || (lang === "mr" ? "दिलेल्या माती व हवामानासाठी थेट अनुकूल पीक आढळले नाही. सिंचन आणि खत व्यवस्थापन तपासा." : "No crop directly matched the input combination. Check irrigation availability.")}
          </div>
        </div>
      `;
    } else {
      crops.forEach((crop, index) => {
        const card = document.createElement("div");
        card.className = "crop-result-card";

        const suitability = crop.suitability || (index === 0 ? (lang === "mr" ? "अत्यंत योग्य" : lang === "hi" ? "अत्यंत उपयुक्त" : "Highly Recommended") : (lang === "mr" ? "योग्य" : lang === "hi" ? "उपयुक्त" : "Recommended"));
        const name = crop.name || crop.id;
        const reason = crop.reason || "";
        const risk = crop.risk || "";
        const evidence = (crop.evidence && crop.evidence.length > 0)
          ? crop.evidence.map(e => `[ICAR p.${e.page || 1}]: ${e.text ? e.text.slice(0, 160) + '...' : ''}`).join(" | ")
          : "";

        card.innerHTML = `
          <div class="crop-card-top">
            <h3 class="crop-card-name">🌱 ${name}</h3>
            <span class="crop-suitability-badge">${suitability}</span>
          </div>
          ${reason ? `<div class="crop-card-reason"><strong>${lang === "mr" ? "शिफारस कारण:" : lang === "hi" ? "सिफारिश का कारण:" : "Recommendation Reason:"}</strong> ${reason}</div>` : ""}
          ${risk ? `<div class="crop-card-risk"><strong>${lang === "mr" ? "हवामान / जोखीम सल्ला:" : lang === "hi" ? "मौसम / जोखिम सलाह:" : "Risk Advisory:"}</strong> ${risk}</div>` : ""}
          ${evidence ? `<div class="crop-evidence-box"><strong>📚 ICAR Knowledge Base:</strong> ${evidence}</div>` : ""}
        `;
        listEl.appendChild(card);
      });
    }

    if (disclaimerEl) {
      disclaimerEl.textContent = result.disclaimer || (lang === "mr"
        ? "टीप: ही शिफारस ICAR मार्गदर्शक तत्त्वांवर आणि नाशिक जिल्ह्याच्या कृषी हवामान स्थितीवर आधारित आहे."
        : "Note: This advice is based on ICAR district agricultural rules and local weather data.");
    }

    container.classList.remove("hidden");
    container.scrollIntoView({ behavior: "smooth", block: "start" });
  }

  function speakResults() {
    if (!_lastResult) return;
    const lang = LanguageSelector.getLanguage();
    const parts = [];
    (_lastResult.crops || []).forEach(c => {
      if (c.name) parts.push(c.name);
      if (c.reason) parts.push(c.reason);
      if (c.risk) parts.push(c.risk);
    });
    if (_lastResult.disclaimer) parts.push(_lastResult.disclaimer);
    const text = parts.join(". ");
    if (text) {
      API.textToSpeech(text, Config.get("voice"), lang).then(data => {
        if (data && data.audio) {
          const snd = new Audio(`data:audio/wav;base64,${data.audio}`);
          snd.play();
        }
      }).catch(err => {
        console.warn("TTS failed:", err);
      });
    }
  }

  function init() {
    _talukaSelect = document.getElementById("crop-taluka-select");
    _monthSelect = document.getElementById("crop-month-select");
    _soilTypeSelect = document.getElementById("crop-soil-type");
    _prevCropSelect = document.getElementById("crop-prev-select");

    // Irrigation toggles
    const irrGroup = document.getElementById("crop-page-irrigation");
    irrGroup?.querySelectorAll(".toggle-btn").forEach(btn => {
      btn.addEventListener("click", () => {
        irrGroup.querySelectorAll(".toggle-btn").forEach(b => b.classList.remove("selected"));
        btn.classList.add("selected");
        _irrigationValue = btn.dataset.value;
      });
    });

    // Form submit
    document.getElementById("dedicated-crop-form")?.addEventListener("submit", handleSubmit);

    // Speak all button
    document.getElementById("btn-crop-speak-all")?.addEventListener("click", speakResults);

    // Re-localize when language changes
    LanguageSelector.onChange(lang => {
      if (_cachedOptions) applyOptions(_cachedOptions, lang);
    });
  }

  return { init, onEnter };
})();


/* ─── DiseasePage (Dedicated Plant Disease Detection View) ──── */
const DiseasePage = (() => {
  let _selectedCrop = "Onion";
  let _selectedFile = null;
  let _lastResult = null;

  function onEnter() {
    // Ready view
  }

  function handleFileSelect(file) {
    clearAlert();
    if (!file) return;

    // Validate type
    const validTypes = ["image/jpeg", "image/png", "image/webp"];
    if (!validTypes.includes(file.type)) {
      showAlert("Please upload a valid image file (JPG, PNG, or WEBP).");
      return;
    }

    // Validate size (max 10MB)
    const maxBytes = 10 * 1024 * 1024;
    if (file.size > maxBytes) {
      showAlert("The selected image is larger than 10MB. Please choose a smaller photo.");
      return;
    }

    _selectedFile = file;

    // Show preview
    const previewContainer = document.getElementById("preview-container");
    const previewImg = document.getElementById("disease-preview-img");
    const promptEl = document.getElementById("dropzone-prompt");
    const filenameEl = document.getElementById("preview-filename");
    const filesizeEl = document.getElementById("preview-filesize");
    const submitBtn = document.getElementById("btn-submit-disease");

    if (previewImg) {
      previewImg.src = URL.createObjectURL(file);
    }
    if (filenameEl) filenameEl.textContent = file.name;
    if (filesizeEl) {
      const mb = (file.size / (1024 * 1024)).toFixed(2);
      filesizeEl.textContent = `${mb} MB`;
    }

    if (promptEl) promptEl.classList.add("hidden");
    if (previewContainer) previewContainer.classList.remove("hidden");
    if (submitBtn) submitBtn.disabled = false;
  }

  function removeFile() {
    _selectedFile = null;
    const fileInput = document.getElementById("disease-file-input");
    if (fileInput) fileInput.value = "";
    const previewContainer = document.getElementById("preview-container");
    const promptEl = document.getElementById("dropzone-prompt");
    const submitBtn = document.getElementById("btn-submit-disease");

    if (previewContainer) previewContainer.classList.add("hidden");
    if (promptEl) promptEl.classList.remove("hidden");
    if (submitBtn) submitBtn.disabled = true;
    clearAlert();
  }

  function showAlert(msg) {
    const alertEl = document.getElementById("disease-form-alert");
    if (alertEl) {
      alertEl.textContent = msg;
      alertEl.className = "form-alert error";
      alertEl.classList.remove("hidden");
    }
  }

  function clearAlert() {
    const alertEl = document.getElementById("disease-form-alert");
    if (alertEl) alertEl.classList.add("hidden");
  }

  async function handleDetect() {
    if (!_selectedFile) {
      showAlert("Please select a leaf photo first.");
      return;
    }
    clearAlert();

    const loadingEl = document.getElementById("disease-loading-state");
    const errorBox = document.getElementById("disease-error-box");
    const resultsContainer = document.getElementById("disease-results-container");
    const submitBtn = document.getElementById("btn-submit-disease");
    const lang = LanguageSelector.getLanguage();

    if (errorBox) errorBox.classList.add("hidden");
    if (resultsContainer) resultsContainer.classList.add("hidden");
    if (loadingEl) loadingEl.classList.remove("hidden");
    if (submitBtn) submitBtn.disabled = true;

    try {
      const result = await DiseaseCheck.checkDisease(_selectedFile, _selectedCrop);
      _lastResult = result;
      if (loadingEl) loadingEl.classList.add("hidden");
      renderResult(result, lang);
    } catch (err) {
      if (loadingEl) loadingEl.classList.add("hidden");
      if (errorBox) {
        const errorMsgEl = document.getElementById("disease-error-message");
        if (errorMsgEl) {
          errorMsgEl.textContent = err.message || (lang === "mr" ? "रोग निदान सर्व्हर जोडता आला नाही." : "Disease detection service is unavailable.");
        }
        errorBox.classList.remove("hidden");
        errorBox.scrollIntoView({ behavior: "smooth", block: "nearest" });
      }
    } finally {
      if (submitBtn) submitBtn.disabled = false;
    }
  }

  function renderResult(result, lang) {
    const container = document.getElementById("disease-results-container");
    const cardEl = document.getElementById("disease-result-card");
    if (!container || !cardEl) return;

    const isHealthy = !!result.is_healthy;
    const status = result.status || "ok";
    const top = (result.top && result.top[0]) ? result.top[0] : null;
    const prob = top ? Math.round(top.probability * 100) : null;
    const diseaseName = (top && top.disease) ? top.disease : (result.crop + " Leaf Inspection");
    const precautions = result.precautions || [];
    const note = result.note || "";
    const message = result.message || "";

    let statusClass = "healthy";
    let statusText = "🌿 Healthy Leaf / निरोगी पान";
    if (!isHealthy) {
      statusClass = status === "not_sure" ? "uncertain" : "diseased";
      statusText = status === "not_sure" ? "⚠️ Low Confidence / अनिश्चित निदान" : "🚨 Disease Detected / रोग आढळला";
    }

    cardEl.innerHTML = `
      <div class="disease-status-banner ${statusClass}">
        <span>${statusText}</span>
        ${prob !== null ? `<span>${prob}% Confidence</span>` : ""}
      </div>

      <h3 class="disease-name-title">${diseaseName}</h3>

      ${prob !== null ? `
        <div class="confidence-box">
          <div style="display:flex; justify-content:space-between; font-size:0.85rem; font-weight:700;">
            <span>AI Model Confidence</span>
            <span>${prob}%</span>
          </div>
          <div class="confidence-bar-wrap">
            <div class="confidence-bar-fill" style="width: ${prob}%;"></div>
          </div>
        </div>
      ` : ""}

      ${message ? `<div style="font-size:var(--fs-base); line-height:1.6; color:var(--c-text);">${message}</div>` : ""}

      ${precautions.length > 0 ? `
        <div class="precautions-section">
          <h4 class="precautions-title">🛡️ ${lang === "mr" ? "उपाययोजना व प्रतिबंधक काळजी:" : lang === "hi" ? "उपाय व सावधानियां:" : "Recommended Precautions & Treatment:"}</h4>
          <ul class="precautions-list">
            ${precautions.map(p => `<li>${p}</li>`).join("")}
          </ul>
        </div>
      ` : ""}

      ${note ? `
        <div class="kvk-advisory-note">
          <strong>ℹ️ KVK Advisory:</strong> ${note}
        </div>
      ` : ""}
    `;

    container.classList.remove("hidden");
    container.scrollIntoView({ behavior: "smooth", block: "start" });
  }

  function speakAdvice() {
    if (!_lastResult) return;
    const lang = LanguageSelector.getLanguage();
    const parts = [];
    if (_lastResult.message) parts.push(_lastResult.message);
    if (_lastResult.precautions && _lastResult.precautions.length > 0) {
      parts.push(_lastResult.precautions.join(". "));
    }
    if (_lastResult.note) parts.push(_lastResult.note);
    const text = parts.join(". ");
    if (text) {
      API.textToSpeech(text, Config.get("voice"), lang).then(data => {
        if (data && data.audio) {
          const snd = new Audio(`data:audio/wav;base64,${data.audio}`);
          snd.play();
        }
      }).catch(err => {
        console.warn("TTS failed:", err);
      });
    }
  }

  function init() {
    // Crop selection pills
    const pills = document.querySelectorAll("#disease-crop-pills .crop-pill");
    pills.forEach(pill => {
      pill.addEventListener("click", () => {
        pills.forEach(p => p.classList.remove("active"));
        pill.classList.add("active");
        _selectedCrop = pill.dataset.crop;
      });
    });

    // Dropzone click & drag
    const dropzone = document.getElementById("disease-dropzone");
    const fileInput = document.getElementById("disease-file-input");

    dropzone?.addEventListener("click", (e) => {
      if (!e.target.closest("#btn-remove-preview")) {
        fileInput?.click();
      }
    });

    fileInput?.addEventListener("change", (e) => {
      if (e.target.files && e.target.files[0]) {
        handleFileSelect(e.target.files[0]);
      }
    });

    dropzone?.addEventListener("dragover", (e) => {
      e.preventDefault();
      dropzone.classList.add("dragover");
    });

    dropzone?.addEventListener("dragleave", () => {
      dropzone.classList.remove("dragover");
    });

    dropzone?.addEventListener("drop", (e) => {
      e.preventDefault();
      dropzone.classList.remove("dragover");
      if (e.dataTransfer && e.dataTransfer.files && e.dataTransfer.files[0]) {
        handleFileSelect(e.dataTransfer.files[0]);
      }
    });

    // Remove preview
    document.getElementById("btn-remove-preview")?.addEventListener("click", (e) => {
      e.stopPropagation();
      removeFile();
    });

    // Detect submit button
    document.getElementById("btn-submit-disease")?.addEventListener("click", handleDetect);

    // Speak button
    document.getElementById("btn-disease-listen")?.addEventListener("click", speakAdvice);

    // Upload another
    document.getElementById("btn-upload-another")?.addEventListener("click", () => {
      removeFile();
      const container = document.getElementById("disease-results-container");
      if (container) container.classList.add("hidden");
      dropzone?.scrollIntoView({ behavior: "smooth", block: "start" });
    });
  }

  return { init, onEnter };
})();


/* ─── Auth Module (Client Authentication & Farmer Profile) ─── */
const Auth = (() => {
  const SESSION_KEY = "agrisaathi_user_session";
  const I18N_AUTH = {
    mr: {
      tagline: "प्रगत शेतीसाठी AI",
      splashLoading: "कृषी AI प्रणाली सुरू होत आहे...",
      splashChecking: "सर्व्हर जोडणी तपासत आहे...",
      splashReady: "सर्व्हर जोडले गेले. डॅशबोर्डवर जात आहे...",
      splashOffline: "सर्व्हर ऑफलाइन आहे (डेमो मोड उपलब्ध). लॉगिन पृष्ठावर जात आहे...",
      skip: "पुढे जा →",
      signInTab: "लॉगिन",
      registerTab: "नवीन नोंदणी",
      identifierLabel: "ईमेल किंवा १०-अंकी मोबाईल नंबर",
      identifierHint: "आपला नोंदणीकृत ईमेल किंवा १०-अंकी मोबाईल नंबर टाका",
      passwordLabel: "पासवर्ड",
      passwordHint: "किमान ६ अक्षरे",
      signInBtn: "लॉगिन करा",
      demoBtn: "⚡ त्वरित डेमो लॉगिन (नाशिक शेतकरी)",
      noAccount: "नवीन शेतकरी आहात?",
      registerHere: "येथे नोंदणी करा",
      fullNameLabel: "शेतकऱ्याचे नाव",
      talukaLabel: "तालुका (नाशिक जिल्हा)",
      confirmPasswordLabel: "पासवर्ड पुन्हा टाका",
      createAccountBtn: "खाते तयार करा",
      haveAccount: "आधीच खाते आहे?",
      signInHere: "लॉगिन करा",
      errEmptyIdentifier: "कृपया ईमेल किंवा मोबाईल नंबर प्रविष्ट करा.",
      errInvalidIdentifier: "कृपया वैध ईमेल किंवा १०-अंकी मोबाईल नंबर प्रविष्ट करा.",
      errPasswordLength: "पासवर्ड किमान ६ अक्षरांचा असणे आवश्यक आहे.",
      errEmptyName: "कृपया आपले संपूर्ण नाव प्रविष्ट करा.",
      errPasswordMismatch: "दोन्ही पासवर्ड जुळत नाहीत.",
      welcomeBack: "🙏 स्वागत आहे",
      registeredSuccess: "खाते यशस्वीरित्या तयार झाले! आपण आता साइन इन केले आहे.",
      loggedOut: "आपण बाहेर पडला आहात.",

      // Navigation
      navDash: "डॅशबोर्ड",
      navCrop: "पीक शिफारस",
      navDisease: "रोग निदान",
      navChat: "कृषी सल्लागार",
      recentChats: "अलीकडील संभाषणे",

      // Dashboard
      dashWelcomeTitle: "तुमचे स्मार्ट कृषी सहाय्यक",
      dashWelcomeDesc: "शेतकऱ्यांना अचूक पीक शिफारस, पानांचे रोग निदान आणि बहुभाषिक व्हॉइस सहाय्याद्वारे योग्य निर्णय घेण्यास मदत.",
      cardCropTitle: "पीक शिफारस",
      cardCropDesc: "मातीतील घटक आणि हवामानानुसार योग्य पिकांची शिफारस मिळवा.",
      btnCrop: "पीक शिफारस मिळवा",
      cardDiseaseTitle: "झाडांचे रोग निदान",
      cardDiseaseDesc: "पानांच्या फोटोवरून रोगाचे निदान करा आणि उपचार सल्ला पहा.",
      btnDisease: "रोग निदान करा",
      cardChatTitle: "AI कृषी सल्लागार चॅटबॉट",
      cardChatDesc: "मराठी, हिंदी किंवा इंग्रजीत शेतीविषयक प्रश्न विचारा आणि व्हॉइस उत्तरे मिळवा.",
      btnChat: "सल्लागार विचारा",

      // Module Pages
      backDashboard: "डॅशबोर्डवर परत जा",
      cropTitle: "पीक शिफारस",
      cropSub: "मातीतील घटक (N, P, K, pH) आणि हवामानानुसार योग्य पिकांची शिफारस मिळवा.",
      diseaseTitle: "झाडांचे रोग निदान",
      diseaseSub: "पानांच्या फोटोवरून रोगाचे अचूक निदान करा आणि उपचार सल्ला मिळवा.",
    },
    hi: {
      tagline: "बेहतर खेती के लिए AI",
      splashLoading: "कृषि AI प्रणाली शुरू हो रही है...",
      splashChecking: "सर्वर कनेक्टिविटी जांची जा रही है...",
      splashReady: "सर्वर कनेक्ट हुआ। डैशबोर्ड पर जा रहे हैं...",
      splashOffline: "सर्वर ऑफलाइन है (डेमो मोड उपलब्ध)। लॉगिन पृष्ठ पर जा रहे हैं...",
      skip: "आगे बढ़ें →",
      signInTab: "साइन इन",
      registerTab: "नया पंजीकरण",
      identifierLabel: "ईमेल या १०-अंकीय मोबाइल नंबर",
      identifierHint: "अपना पंजीकृत ईमेल या १०-अंकीय मोबाइल नंबर दर्ज करें",
      passwordLabel: "पासवर्ड",
      passwordHint: "न्यूनतम ६ वर्ण",
      signInBtn: "साइन इन करें",
      demoBtn: "⚡ त्वरित डेमो लॉगिन (नासिक किसान)",
      noAccount: "नए किसान हैं?",
      registerHere: "यहाँ पंजीकरण करें",
      fullNameLabel: "किसान का नाम",
      talukaLabel: "तालुका (नासिक जिला)",
      confirmPasswordLabel: "पासवर्ड दोबारा दर्ज करें",
      createAccountBtn: "खाता बनाएं",
      haveAccount: "पहले से खाता है?",
      signInHere: "साइन इन करें",
      errEmptyIdentifier: "कृपया ईमेल या मोबाइल नंबर दर्ज करें।",
      errInvalidIdentifier: "कृपया एक वैध ईमेल या १०-अंकीय मोबाइल नंबर दर्ज करें।",
      errPasswordLength: "पासवर्ड कम से कम ६ अक्षरों का होना चाहिए।",
      errEmptyName: "कृपया अपना पूरा नाम दर्ज करें।",
      errPasswordMismatch: "दोनों पासवर्ड मेल नहीं खाते।",
      welcomeBack: "🙏 स्वागत है",
      registeredSuccess: "खाता सफलतापूर्वक बनाया गया! आप अब साइन इन हैं।",
      loggedOut: "आप लॉग आउट हो गए हैं।",

      // Navigation
      navDash: "डैशबोर्ड",
      navCrop: "फसल सिफारिश",
      navDisease: "रोग पहचान",
      navChat: "कृषि सहायक",
      recentChats: "हाल की बातचीत",

      // Dashboard
      dashWelcomeTitle: "आपका स्मार्ट कृषि सहायक",
      dashWelcomeDesc: "किसानों को फसल सिफारिश, पत्तियों के रोग निदान और बहुभाषी वॉयस सहायता से बेहतर निर्णय लेने में मदद।",
      cardCropTitle: "फसल सिफारिश",
      cardCropDesc: "मिट्टी के पोषक तत्वों और मौसम के अनुसार उपयुक्त फसलें जानें।",
      btnCrop: "फसल सिफारिश प्राप्त करें",
      cardDiseaseTitle: "पौध रोग पहचान",
      cardDiseaseDesc: "पत्तियों के फोटो से रोग की पहचान करें और उपचार सलाह देखें।",
      btnDisease: "रोग की पहचान करें",
      cardChatTitle: "AI कृषि सलाहकार चैटबॉट",
      cardChatDesc: "मराठी, हिंदी या अंग्रेजी में कृषि प्रश्न पूछें और वॉयस उत्तर प्राप्त करें।",
      btnChat: "सहायक से पूछें",

      // Module Pages
      backDashboard: "डैशबोर्ड पर वापस जाएं",
      cropTitle: "फसल सिफारिश",
      cropSub: "मिट्टी के पोषक तत्वों (N, P, K, pH) और मौसम के अनुसार उपयुक्त फसलें जानें।",
      diseaseTitle: "पौध रोग पहचान",
      diseaseSub: "पत्तियों के फोटो से रोग की सही पहचान करें और उपचार सलाह प्राप्त करें।",
    },
    en: {
      tagline: "AI for Better Farming",
      splashLoading: "Initializing agricultural intelligence...",
      splashChecking: "Verifying backend connectivity...",
      splashReady: "Backend online. Navigating to dashboard...",
      splashOffline: "Backend offline (Demo Mode available). Redirecting to login...",
      skip: "Continue to App →",
      signInTab: "Sign In",
      registerTab: "Register",
      identifierLabel: "Email or 10-Digit Mobile Number",
      identifierHint: "Enter your registered email or 10-digit mobile number",
      passwordLabel: "Password",
      passwordHint: "Must be at least 6 characters",
      signInBtn: "Sign In",
      demoBtn: "⚡ Quick Demo Login (Nashik Farmer)",
      noAccount: "New to AgriSaathi?",
      registerHere: "Register here",
      fullNameLabel: "Farmer Full Name",
      talukaLabel: "Taluka (Nashik District)",
      confirmPasswordLabel: "Confirm Password",
      createAccountBtn: "Create Account",
      haveAccount: "Already have an account?",
      signInHere: "Sign In",
      errEmptyIdentifier: "Please enter your email or mobile number.",
      errInvalidIdentifier: "Please enter a valid email or 10-digit mobile number.",
      errPasswordLength: "Password must be at least 6 characters.",
      errEmptyName: "Please enter your full name.",
      errPasswordMismatch: "Passwords do not match.",
      welcomeBack: "🙏 Welcome",
      registeredSuccess: "Account created successfully! You are now logged in.",
      loggedOut: "Logged out successfully.",

      // Navigation
      navDash: "Dashboard",
      navCrop: "Crop Recommendation",
      navDisease: "Disease Detection",
      navChat: "AI Assistant",
      // Shorter labels for the phone bottom bar (mr/hi labels already fit)
      mobDash: "Home",
      mobCrop: "Crops",
      mobDisease: "Disease",
      mobChat: "Assistant",
      recentChats: "Recent Chats",

      // Dashboard
      dashWelcomeTitle: "Your Smart Farming Assistant",
      dashWelcomeDesc: "AgriSaathi helps farmers make better farming decisions through intelligent crop recommendations, leaf disease detection, and multilingual voice advisory in Marathi, Hindi, and English.",
      cardCropTitle: "Crop Recommendation",
      cardCropDesc: "Find suitable crops using soil nutrients and weather conditions.",
      btnCrop: "Recommend Crop",
      cardDiseaseTitle: "Plant Disease Detection",
      cardDiseaseDesc: "Identify plant diseases from leaf images and view treatment advice.",
      btnDisease: "Detect Disease",
      cardChatTitle: "AI Agricultural Chatbot",
      cardChatDesc: "Ask farming questions in Marathi, Hindi, or English using text or voice.",
      btnChat: "Ask AI Assistant",

      // Module Pages
      backDashboard: "Back to Dashboard",
      cropTitle: "Crop Recommendation",
      cropSub: "Find suitable crops using soil nutrients, pH, and environmental conditions.",
      diseaseTitle: "Plant Disease Detection",
      diseaseSub: "Identify plant diseases from leaf images and view treatment advice.",
    }
  };

  let _currentUser = null;

  function _getText(key) {
    const lang = LanguageSelector.getLanguage() || "mr";
    const dict = I18N_AUTH[lang] || I18N_AUTH.mr;
    return dict[key] || I18N_AUTH.en[key] || "";
  }

  function _loadSession() {
    try {
      const data = localStorage.getItem(SESSION_KEY);
      if (data) {
        _currentUser = JSON.parse(data);
      }
    } catch (e) {
      console.warn("Failed to read user session:", e);
      _currentUser = null;
    }
  }

  function _saveSession(user) {
    _currentUser = user;
    try {
      localStorage.setItem(SESSION_KEY, JSON.stringify(user));
    } catch (e) {
      console.warn("Failed to store user session:", e);
    }
  }

  function _clearSession() {
    _currentUser = null;
    try {
      localStorage.removeItem(SESSION_KEY);
    } catch (e) {
      console.warn("Failed to clear session:", e);
    }
  }

  function isValidEmail(str) {
    return /^[^\s@]+@[^\s@]+\.[^\s@]+$/.test(str.trim());
  }

  function isValidPhone(str) {
    const cleaned = str.replace(/[\s\-\+]/g, "");
    return /^[6-9]\d{9}$/.test(cleaned);
  }

  function showAlert(msg, isError = true) {
    const alertEl = document.getElementById("auth-alert");
    if (!alertEl) return;
    alertEl.textContent = msg;
    alertEl.className = `auth-alert ${isError ? "error" : "success"}`;
    alertEl.classList.remove("hidden");
  }

  function clearAlert() {
    const alertEl = document.getElementById("auth-alert");
    if (!alertEl) return;
    alertEl.textContent = "";
    alertEl.classList.add("hidden");
  }

  function updateTexts() {
    const lang = LanguageSelector.getLanguage() || "mr";
    const dict = I18N_AUTH[lang] || I18N_AUTH.mr;

    // Splash texts
    const elSub = document.getElementById("splash-sub-text");
    if (elSub) elSub.textContent = (lang === "en") ? "AI Assistant" : "एग्रीसाथी";
    const elTag = document.getElementById("splash-tagline");
    if (elTag) elTag.textContent = dict.tagline;
    const elSkip = document.getElementById("splash-skip-btn");
    if (elSkip) elSkip.textContent = dict.skip;

    // Auth screen texts
    const authSub = document.getElementById("auth-subtitle");
    if (authSub) authSub.textContent = dict.tagline;
    const tabLoginText = document.getElementById("tab-login-text");
    if (tabLoginText) tabLoginText.textContent = dict.signInTab;
    const tabRegText = document.getElementById("tab-register-text");
    if (tabRegText) tabRegText.textContent = dict.registerTab;

    const lblLoginIdent = document.getElementById("lbl-login-identifier");
    if (lblLoginIdent) lblLoginIdent.textContent = dict.identifierLabel;
    const hintLoginIdent = document.getElementById("hint-login-identifier");
    if (hintLoginIdent) hintLoginIdent.textContent = dict.identifierHint;

    const lblLoginPass = document.getElementById("lbl-login-password");
    if (lblLoginPass) lblLoginPass.textContent = dict.passwordLabel;
    const hintLoginPass = document.getElementById("hint-login-password");
    if (hintLoginPass) hintLoginPass.textContent = dict.passwordHint;

    const btnLoginSubmit = document.getElementById("btn-login-submit");
    if (btnLoginSubmit) btnLoginSubmit.textContent = dict.signInBtn;
    const btnDemo = document.getElementById("btn-demo-login");
    if (btnDemo) btnDemo.textContent = dict.demoBtn;

    const txtNoAccount = document.getElementById("txt-no-account");
    if (txtNoAccount) txtNoAccount.textContent = dict.noAccount;
    const linkToReg = document.getElementById("link-to-register");
    if (linkToReg) linkToReg.textContent = dict.registerHere;

    const lblRegName = document.getElementById("lbl-reg-name");
    if (lblRegName) lblRegName.textContent = dict.fullNameLabel;
    const lblRegIdent = document.getElementById("lbl-reg-identifier");
    if (lblRegIdent) lblRegIdent.textContent = dict.identifierLabel;
    const lblRegTaluka = document.getElementById("lbl-reg-taluka");
    if (lblRegTaluka) lblRegTaluka.textContent = dict.talukaLabel;
    const lblRegPass = document.getElementById("lbl-reg-password");
    if (lblRegPass) lblRegPass.textContent = dict.passwordLabel;
    const lblRegConf = document.getElementById("lbl-reg-confirm-password");
    if (lblRegConf) lblRegConf.textContent = dict.confirmPasswordLabel;

    const btnRegSubmit = document.getElementById("btn-reg-submit");
    if (btnRegSubmit) btnRegSubmit.textContent = dict.createAccountBtn;
    const txtHaveAccount = document.getElementById("txt-have-account");
    if (txtHaveAccount) txtHaveAccount.textContent = dict.haveAccount;
    const linkToLogin = document.getElementById("link-to-login");
    if (linkToLogin) linkToLogin.textContent = dict.signInHere;

    // Header & Navigation texts
    const logoutBtn = document.getElementById("logout-btn");
    if (logoutBtn) logoutBtn.textContent = (lang === "mr") ? "🚪 बाहेर पडा" : (lang === "hi") ? "🚪 लॉगआउट" : "🚪 Logout";

    const topNavDash = document.getElementById("top-nav-dash");
    if (topNavDash) topNavDash.textContent = dict.navDash;
    const topNavCrop = document.getElementById("top-nav-crop");
    if (topNavCrop) topNavCrop.textContent = dict.navCrop;
    const topNavDisease = document.getElementById("top-nav-disease");
    if (topNavDisease) topNavDisease.textContent = dict.navDisease;
    const topNavChat = document.getElementById("top-nav-chat");
    if (topNavChat) topNavChat.textContent = dict.navChat;

    const sideNavDash = document.getElementById("lbl-nav-dash");
    if (sideNavDash) sideNavDash.textContent = dict.navDash;
    const sideNavCrop = document.getElementById("lbl-nav-crop");
    if (sideNavCrop) sideNavCrop.textContent = dict.navCrop;
    const sideNavDisease = document.getElementById("lbl-nav-disease");
    if (sideNavDisease) sideNavDisease.textContent = dict.navDisease;
    const sideNavChat = document.getElementById("lbl-nav-chat");
    if (sideNavChat) sideNavChat.textContent = dict.navChat;

    const mobLblDash = document.getElementById("mob-lbl-dash");
    if (mobLblDash) mobLblDash.textContent = dict.mobDash || dict.navDash;
    const mobLblCrop = document.getElementById("mob-lbl-crop");
    if (mobLblCrop) mobLblCrop.textContent = dict.mobCrop || dict.navCrop;
    const mobLblDisease = document.getElementById("mob-lbl-disease");
    if (mobLblDisease) mobLblDisease.textContent = dict.mobDisease || dict.navDisease;
    const mobLblChat = document.getElementById("mob-lbl-chat");
    if (mobLblChat) mobLblChat.textContent = dict.mobChat || dict.navChat;

    const sideHistTitle = document.getElementById("lbl-sidebar-history-title");
    if (sideHistTitle) sideHistTitle.textContent = dict.recentChats;

    // Dashboard texts
    const dashWelcome = document.getElementById("dash-welcome-title");
    if (dashWelcome) dashWelcome.textContent = dict.dashWelcomeTitle;
    const dashDesc = document.getElementById("dash-welcome-desc");
    if (dashDesc) dashDesc.textContent = dict.dashWelcomeDesc;

    const cardCropTitle = document.getElementById("card-crop-title");
    if (cardCropTitle) cardCropTitle.textContent = dict.cardCropTitle;
    const cardCropDesc = document.getElementById("card-crop-desc");
    if (cardCropDesc) cardCropDesc.textContent = dict.cardCropDesc;
    const txtBtnCrop = document.getElementById("txt-btn-crop");
    if (txtBtnCrop) txtBtnCrop.textContent = dict.btnCrop;

    const cardDiseaseTitle = document.getElementById("card-disease-title");
    if (cardDiseaseTitle) cardDiseaseTitle.textContent = dict.cardDiseaseTitle;
    const cardDiseaseDesc = document.getElementById("card-disease-desc");
    if (cardDiseaseDesc) cardDiseaseDesc.textContent = dict.cardDiseaseDesc;
    const txtBtnDisease = document.getElementById("txt-btn-disease");
    if (txtBtnDisease) txtBtnDisease.textContent = dict.btnDisease;

    const cardChatTitle = document.getElementById("card-chat-title");
    if (cardChatTitle) cardChatTitle.textContent = dict.cardChatTitle;
    const cardChatDesc = document.getElementById("card-chat-desc");
    if (cardChatDesc) cardChatDesc.textContent = dict.cardChatDesc;
    const txtBtnChat = document.getElementById("txt-btn-chat");
    if (txtBtnChat) txtBtnChat.textContent = dict.btnChat;

    // Module Page Titles and Back buttons
    const txtCropBack = document.getElementById("txt-crop-back");
    if (txtCropBack) txtCropBack.textContent = dict.backDashboard;
    const txtDiseaseBack = document.getElementById("txt-disease-back");
    if (txtDiseaseBack) txtDiseaseBack.textContent = dict.backDashboard;
    const txtChatBack = document.getElementById("txt-chat-back");
    if (txtChatBack) txtChatBack.textContent = dict.backDashboard;

    const cropPageTitle = document.getElementById("crop-page-title");
    if (cropPageTitle) cropPageTitle.textContent = dict.cropTitle;
    const cropPageSubtitle = document.getElementById("crop-page-subtitle");
    if (cropPageSubtitle) cropPageSubtitle.textContent = dict.cropSub;

    const diseasePageTitle = document.getElementById("disease-page-title");
    if (diseasePageTitle) diseasePageTitle.textContent = dict.diseaseTitle;
    const diseasePageSubtitle = document.getElementById("disease-page-subtitle");
    if (diseasePageSubtitle) diseasePageSubtitle.textContent = dict.diseaseSub;

    // Greeting
    const user = getCurrentUser();
    const greetingEl = document.getElementById("dash-farmer-greeting");
    if (greetingEl) {
      const name = user?.name || "Farmer";
      greetingEl.textContent = (lang === "mr") ? `स्वागत आहे, ${name}!` : (lang === "hi") ? `स्वागत है, ${name}!` : `Welcome back, ${name}!`;
    }
  }

  function showLoginTab() {
    clearAlert();
    document.getElementById("tab-login")?.classList.add("active");
    document.getElementById("tab-login")?.setAttribute("aria-selected", "true");
    document.getElementById("tab-register")?.classList.remove("active");
    document.getElementById("tab-register")?.setAttribute("aria-selected", "false");
    document.getElementById("login-form")?.classList.remove("hidden");
    document.getElementById("register-form")?.classList.add("hidden");
  }

  function showRegisterTab() {
    clearAlert();
    document.getElementById("tab-register")?.classList.add("active");
    document.getElementById("tab-register")?.setAttribute("aria-selected", "true");
    document.getElementById("tab-login")?.classList.remove("active");
    document.getElementById("tab-login")?.setAttribute("aria-selected", "false");
    document.getElementById("register-form")?.classList.remove("hidden");
    document.getElementById("login-form")?.classList.add("hidden");
  }

  function setupPasswordToggle(inputId, toggleBtnId) {
    const input = document.getElementById(inputId);
    const btn = document.getElementById(toggleBtnId);
    if (!input || !btn) return;

    btn.addEventListener("click", () => {
      const isPassword = input.type === "password";
      input.type = isPassword ? "text" : "password";
      btn.textContent = isPassword ? "🙈" : "👁️";
      btn.setAttribute("aria-label", isPassword ? "Hide password" : "Show password");
    });
  }

  function completeLogin(user) {
    _saveSession(user);
    const authScreen = document.getElementById("auth-screen");
    const layoutContainer = document.getElementById("layout-container");
    const userDisplay = document.getElementById("user-display-name");
    const sideUserDisplay = document.getElementById("sidebar-user-name");

    if (authScreen) authScreen.style.display = "none";
    if (layoutContainer) layoutContainer.style.display = "flex";
    const name = user.name || user.identifier.split("@")[0] || "Farmer";
    if (userDisplay) userDisplay.textContent = name;
    if (sideUserDisplay) sideUserDisplay.textContent = name;

    updateTexts();
    Router.navigate("dashboard");
    Toast.show(`${_getText("welcomeBack")}, ${name}!`, "success", 4000);
  }

  function logout() {
    _clearSession();
    const authScreen = document.getElementById("auth-screen");
    const layoutContainer = document.getElementById("layout-container");

    if (layoutContainer) layoutContainer.style.display = "none";
    if (authScreen) authScreen.style.display = "flex";
    window.location.hash = "";
    showLoginTab();
    Toast.show(_getText("loggedOut"), "info", 3000);
  }

  function init() {
    _loadSession();

    // Password toggles
    setupPasswordToggle("login-password", "toggle-login-password");
    setupPasswordToggle("reg-password", "toggle-reg-password");
    setupPasswordToggle("reg-confirm-password", "toggle-reg-confirm-password");

    // Tab buttons
    document.getElementById("tab-login")?.addEventListener("click", showLoginTab);
    document.getElementById("tab-register")?.addEventListener("click", showRegisterTab);
    document.getElementById("link-to-register")?.addEventListener("click", (e) => {
      e.preventDefault();
      showRegisterTab();
    });
    document.getElementById("link-to-login")?.addEventListener("click", (e) => {
      e.preventDefault();
      showLoginTab();
    });

    // Language switcher buttons on login screen
    document.querySelectorAll(".auth-lang-btn").forEach((btn) => {
      btn.addEventListener("click", () => {
        LanguageSelector.setLanguage(btn.dataset.lang);
      });
    });

    // Logout buttons (Header & Sidebar)
    document.getElementById("logout-btn")?.addEventListener("click", () => {
      logout();
    });
    document.getElementById("sidebar-logout-btn")?.addEventListener("click", () => {
      logout();
    });

    // Demo Login button
    document.getElementById("btn-demo-login")?.addEventListener("click", () => {
      const demoUser = {
        name: "Subhash Patil (सुभाष पाटील)",
        identifier: "9822012345",
        taluka: "Niphad",
        role: "farmer",
        loginTime: new Date().toISOString()
      };
      completeLogin(demoUser);
    });

    // Login Form Submit
    const loginForm = document.getElementById("login-form");
    loginForm?.addEventListener("submit", (e) => {
      e.preventDefault();
      clearAlert();

      const identInput = document.getElementById("login-identifier");
      const passInput = document.getElementById("login-password");
      const ident = identInput?.value.trim() || "";
      const pass = passInput?.value || "";

      if (!ident) {
        showAlert(_getText("errEmptyIdentifier"));
        identInput?.focus();
        return;
      }
      if (!isValidEmail(ident) && !isValidPhone(ident)) {
        showAlert(_getText("errInvalidIdentifier"));
        identInput?.focus();
        return;
      }
      if (pass.length < 6) {
        showAlert(_getText("errPasswordLength"));
        passInput?.focus();
        return;
      }

      // Successful validation: create authenticated session
      const user = {
        name: isValidPhone(ident) ? `Farmer (${ident.slice(-4)})` : ident.split("@")[0],
        identifier: ident,
        taluka: "Nashik",
        role: "farmer",
        loginTime: new Date().toISOString()
      };
      completeLogin(user);
    });

    // Registration Form Submit
    const regForm = document.getElementById("register-form");
    regForm?.addEventListener("submit", (e) => {
      e.preventDefault();
      clearAlert();

      const nameInput = document.getElementById("reg-name");
      const identInput = document.getElementById("reg-identifier");
      const talukaInput = document.getElementById("reg-taluka");
      const passInput = document.getElementById("reg-password");
      const confPassInput = document.getElementById("reg-confirm-password");

      const name = nameInput?.value.trim() || "";
      const ident = identInput?.value.trim() || "";
      const taluka = talukaInput?.value || "Nashik";
      const pass = passInput?.value || "";
      const confPass = confPassInput?.value || "";

      if (!name) {
        showAlert(_getText("errEmptyName"));
        nameInput?.focus();
        return;
      }
      if (!isValidEmail(ident) && !isValidPhone(ident)) {
        showAlert(_getText("errInvalidIdentifier"));
        identInput?.focus();
        return;
      }
      if (pass.length < 6) {
        showAlert(_getText("errPasswordLength"));
        passInput?.focus();
        return;
      }
      if (pass !== confPass) {
        showAlert(_getText("errPasswordMismatch"));
        confPassInput?.focus();
        return;
      }

      const user = {
        name: name,
        identifier: ident,
        taluka: taluka,
        role: "farmer",
        loginTime: new Date().toISOString()
      };
      completeLogin(user);
    });

    // Listen to language changes
    LanguageSelector.onChange(() => {
      updateTexts();
    });
    updateTexts();
  }

  function getCurrentUser() {
    return _currentUser;
  }

  function isLoggedIn() {
    return !!_currentUser;
  }

  return { init, getCurrentUser, isLoggedIn, completeLogin, logout, updateTexts };
})();


/* ─── Splash Screen Coordinator ───────────────────────────── */
const SplashScreen = (() => {
  let _hasNavigated = false;

  function dismiss() {
    if (_hasNavigated) return;
    _hasNavigated = true;

    const splash = document.getElementById("splash-screen");
    const authScreen = document.getElementById("auth-screen");
    const layoutContainer = document.getElementById("layout-container");

    if (splash) {
      splash.classList.add("fade-out");
      setTimeout(() => {
        splash.style.display = "none";
      }, 500);
    }

    // If user is already authenticated in session, navigate directly to dashboard
    if (Auth.isLoggedIn()) {
      if (layoutContainer) layoutContainer.style.display = "flex";
      const user = Auth.getCurrentUser();
      const userDisplay = document.getElementById("user-display-name");
      const sideUserDisplay = document.getElementById("sidebar-user-name");
      if (userDisplay && user) {
        userDisplay.textContent = user.name || "Farmer";
      }
      if (sideUserDisplay && user) {
        sideUserDisplay.textContent = user.name || "Farmer";
      }
      const initialRoute = window.location.hash.replace(/^#\/?/, "") || "dashboard";
      Router.navigate(initialRoute);
    } else {
      if (authScreen) authScreen.style.display = "flex";
      setTimeout(() => {
        document.getElementById("login-identifier")?.focus();
      }, 550);
    }
  }

  async function init() {
    const statusEl = document.getElementById("splash-status");
    const skipBtn = document.getElementById("splash-skip-btn");

    skipBtn?.addEventListener("click", () => {
      dismiss();
    });

    // Update status text
    const lang = LanguageSelector.getLanguage() || "mr";
    if (statusEl) {
      statusEl.textContent = (lang === "mr")
        ? "कृषी AI प्रणाली सुरू होत आहे..."
        : (lang === "hi")
        ? "कृषि AI प्रणाली शुरू हो रही है..."
        : "Initializing agricultural intelligence...";
    }

    // Safety timeout: dismiss after 2.2 seconds maximum
    const fallbackTimer = setTimeout(() => {
      dismiss();
    }, 2200);

    // Try checking backend health with a fast timeout
    try {
      const healthPromise = API.healthCheck();
      const timeoutPromise = new Promise((_, reject) =>
        setTimeout(() => reject(new Error("Health check timeout")), 1200)
      );

      await Promise.race([healthPromise, timeoutPromise]);
      if (statusEl) {
        statusEl.textContent = (lang === "mr")
          ? "सर्व्हर जोडले गेले. डॅशबोर्डवर जात आहे..."
          : (lang === "hi")
          ? "सर्वर कनेक्ट हुआ। डैशबोर्ड पर जा रहे हैं..."
          : "Backend online. Navigating to dashboard...";
      }
    } catch {
      if (statusEl) {
        statusEl.textContent = (lang === "mr")
          ? "प्रणाली सज्ज आहे. डॅशबोर्डवर जात आहे..."
          : (lang === "hi")
          ? "प्रणाली तैयार है। डैशबोर्ड पर जा रहे हैं..."
          : "System ready. Navigating to dashboard...";
      }
    }

    // Graceful brief pause so the spinner and branding are nicely seen
    setTimeout(() => {
      clearTimeout(fallbackTimer);
      dismiss();
    }, 1500);
  }

  return { init, dismiss };
})();


/* ─── Boot Sequence ───────────────────────────────────────── */
document.addEventListener("DOMContentLoaded", () => {
  App.init();
  Auth.init();
  Router.init();
  CropPage.init();
  DiseasePage.init();
  SplashScreen.init();
});


