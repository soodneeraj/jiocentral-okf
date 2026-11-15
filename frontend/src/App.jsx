import React, { useCallback, useEffect, useMemo, useRef, useState } from "react";

const API = "";

const PROGRESS_STEPS = [
  "Understanding your question…",
  "Searching document text…",
  "Searching figures and diagrams…",
  "Assembling evidence…",
  "Generating grounded answer…",
];

function imageUrl(path) {
  if (!path) return "";
  if (path.startsWith("http") || path.startsWith("/")) return path;
  return `/static/bundle/${path}`;
}

function formatDuration(ms) {
  if (ms == null || Number.isNaN(ms)) return "";
  if (ms < 1000) return `${Math.round(ms)} ms`;
  const sec = ms / 1000;
  if (sec < 60) return `${sec.toFixed(1)} s`;
  const m = Math.floor(sec / 60);
  const s = (sec % 60).toFixed(1);
  return `${m}m ${s}s`;
}

export default function App() {
  const [docs, setDocs] = useState([]);
  const [messages, setMessages] = useState([]);
  const [input, setInput] = useState("");
  const [busy, setBusy] = useState(false);
  const [progressElapsedMs, setProgressElapsedMs] = useState(0);
  const [progressStep, setProgressStep] = useState(0);
  const [conversationId, setConversationId] = useState(null);
  const [lightbox, setLightbox] = useState(null);
  const [job, setJob] = useState(null);
  const [health, setHealth] = useState(null);
  const messagesEndRef = useRef(null);
  const tickRef = useRef(null);
  const stepRef = useRef(null);

  const refreshDocs = useCallback(async () => {
    try {
      const res = await fetch(`${API}/api/v1/documents`);
      const data = await res.json();
      setDocs(data.documents || []);
    } catch (e) {
      console.error(e);
    }
  }, []);

  useEffect(() => {
    refreshDocs();
    fetch(`${API}/api/v1/health`)
      .then((r) => r.json())
      .then(setHealth)
      .catch(() => {});
  }, [refreshDocs]);

  useEffect(() => {
    if (!job?.id || job.state === "done" || job.state === "error") return;
    const t = setInterval(async () => {
      const res = await fetch(`${API}/api/v1/jobs/${job.id}`);
      const data = await res.json();
      setJob(data);
      if (data.state === "done" || data.state === "error") {
        refreshDocs();
      }
    }, 2000);
    return () => clearInterval(t);
  }, [job, refreshDocs]);

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages, busy, progressElapsedMs]);

  function clearProgressTimers() {
    if (tickRef.current) {
      clearInterval(tickRef.current);
      tickRef.current = null;
    }
    if (stepRef.current) {
      clearInterval(stepRef.current);
      stepRef.current = null;
    }
  }

  function startProgressTimers() {
    clearProgressTimers();
    const started = performance.now();
    setProgressElapsedMs(0);
    setProgressStep(0);
    tickRef.current = setInterval(() => {
      setProgressElapsedMs(performance.now() - started);
    }, 200);
    stepRef.current = setInterval(() => {
      setProgressStep((s) => Math.min(s + 1, PROGRESS_STEPS.length - 1));
    }, 4500);
  }

  async function onUpload(e) {
    const files = e.target.files;
    if (!files?.length) return;
    const fd = new FormData();
    for (const f of files) fd.append("files", f);
    setBusy(true);
    try {
      const res = await fetch(
        `${API}/api/v1/documents/upload?process=true&do_index=true`,
        { method: "POST", body: fd }
      );
      const data = await res.json();
      setJob(data.job);
      await refreshDocs();
    } finally {
      setBusy(false);
      e.target.value = "";
    }
  }

  async function send() {
    const q = input.trim();
    if (!q || busy) return;
    setInput("");
    setMessages((m) => [...m, { role: "user", content: q }]);
    setBusy(true);
    startProgressTimers();
    const t0 = performance.now();
    try {
      const history = messages
        .filter((m) => m.role === "user" || m.role === "assistant")
        .map((m) => ({ role: m.role, content: m.content }));
      const res = await fetch(`${API}/api/v1/chat/query`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          query: q,
          conversation_id: conversationId,
          history,
          top_k: 10,
        }),
      });
      const data = await res.json();
      const elapsedMs = performance.now() - t0;
      if (data.conversation_id) setConversationId(data.conversation_id);
      setMessages((m) => [
        ...m,
        {
          role: "assistant",
          content: data.answer || "",
          sources: data.sources || [],
          images: data.images || [],
          evidence_map: data.evidence_map || [],
          warnings: data.warnings || [],
          intent: data.intent,
          elapsedMs,
        },
      ]);
    } catch (err) {
      const elapsedMs = performance.now() - t0;
      setMessages((m) => [
        ...m,
        {
          role: "assistant",
          content: `Error: ${err.message}`,
          elapsedMs,
        },
      ]);
    } finally {
      clearProgressTimers();
      setBusy(false);
      setProgressElapsedMs(0);
      setProgressStep(0);
    }
  }

  function newChat() {
    clearProgressTimers();
    setMessages([]);
    setConversationId(null);
    setBusy(false);
    setProgressElapsedMs(0);
  }

  const latestImages = useMemo(() => {
    for (let i = messages.length - 1; i >= 0; i--) {
      if (messages[i].images?.length) return messages[i].images;
    }
    return [];
  }, [messages]);

  return (
    <div className="app">
      <aside className="sidebar">
        <div className="brand">OKF Multimodal RAG</div>
        <div className="sub">On-premise document intelligence</div>
        <button onClick={newChat} style={{ width: "100%", marginBottom: 12 }}>
          New conversation
        </button>
        <div className="upload-row">
          <label className="btn secondary" style={{ textAlign: "center" }}>
            Upload PDFs
            <input
              type="file"
              accept="application/pdf"
              multiple
              hidden
              onChange={onUpload}
            />
          </label>
          {job && (
            <div className="job-item">
              Job {job.id?.slice?.(0, 8) || job.id}:{" "}
              <span className="status">{job.state}</span>
              <div>{job.message}</div>
            </div>
          )}
        </div>
        <div className="sub">Documents</div>
        {docs.map((d) => (
          <div className="doc-item" key={d.document_id}>
            <div>{d.original_filename || d.name}</div>
            <div className="status">
              {d.processing_status} · type {d.document_type} · {d.page_count}p
            </div>
          </div>
        ))}
        {health && (
          <div className="doc-item" style={{ marginTop: 16 }}>
            <div>Milvus: {String(health.milvus)}</div>
            <div className="status">
              Models: {health.models?.ollama_reachable ? "reachable" : "down"}
            </div>
          </div>
        )}
      </aside>

      <main className="main">
        <div className="messages">
          {messages.length === 0 && !busy && (
            <div className="bubble ai">
              Ask questions across your authorized PDF repository. Answers include
              page citations and original figures when relevant.
            </div>
          )}
          {messages.map((m, idx) => (
            <React.Fragment key={idx}>
              <div className={`bubble ${m.role === "user" ? "user" : "ai"}`}>
                {m.content}
                {m.role === "assistant" && m.elapsedMs != null && (
                  <div className="meta-time">
                    Response time: {formatDuration(m.elapsedMs)}
                  </div>
                )}
                {m.intent && (
                  <div className="status" style={{ marginTop: 8 }}>
                    Intent: {m.intent}
                  </div>
                )}
                {m.warnings?.length > 0 && (
                  <div className="warn">Warnings: {m.warnings.join(", ")}</div>
                )}
              </div>
              {m.role === "assistant" && (m.sources?.length > 0 || m.images?.length > 0) && (
                <div className="bubble evidence">
                  <strong>Document-supported evidence</strong>
                  {(m.sources || []).map((s, i) => (
                    <div key={i}>
                      {s.source} {s.source_loc}
                      {s.page != null ? ` (page ${s.page})` : ""}
                    </div>
                  ))}
                  <div className="thumbs" style={{ marginTop: 8 }}>
                    {(m.images || []).map((img, i) => (
                      <img
                        key={i}
                        src={imageUrl(img.image_path)}
                        alt={img.caption || "figure"}
                        onClick={() => setLightbox(imageUrl(img.image_path))}
                      />
                    ))}
                  </div>
                </div>
              )}
            </React.Fragment>
          ))}

          {busy && (
            <div className="bubble progress" aria-live="polite" aria-busy="true">
              <div className="progress-row">
                <span className="spinner" aria-hidden="true" />
                <div>
                  <div className="progress-title">Generating response…</div>
                  <div className="progress-step">{PROGRESS_STEPS[progressStep]}</div>
                  <div className="progress-timer">
                    Elapsed: {formatDuration(progressElapsedMs)}
                  </div>
                </div>
              </div>
              <div className="progress-bar">
                <div
                  className="progress-bar-fill"
                  style={{
                    width: `${Math.min(92, 12 + progressStep * 18 + (progressElapsedMs % 3000) / 120)}%`,
                  }}
                />
              </div>
            </div>
          )}
          <div ref={messagesEndRef} />
        </div>
        <div className="composer">
          <textarea
            value={input}
            placeholder={busy ? "Waiting for response…" : "Ask a question about your documents…"}
            disabled={busy}
            onChange={(e) => setInput(e.target.value)}
            onKeyDown={(e) => {
              if (e.key === "Enter" && !e.shiftKey) {
                e.preventDefault();
                send();
              }
            }}
          />
          <button onClick={send} disabled={busy || !input.trim()}>
            {busy ? "Working…" : "Send"}
          </button>
        </div>
      </main>

      <aside className="rail">
        <div className="sub">Relevant images</div>
        <div className="thumbs">
          {latestImages.map((img, i) => (
            <div key={i}>
              <img
                src={imageUrl(img.image_path)}
                alt=""
                onClick={() => setLightbox(imageUrl(img.image_path))}
              />
              <div className="status">
                p{img.page} · {img.source || ""}
              </div>
            </div>
          ))}
        </div>
        <div className="sub" style={{ marginTop: 16 }}>
          Open source PDF at cited page via your PDF viewer using page numbers
          shown in evidence.
        </div>
      </aside>

      {lightbox && (
        <div className="lightbox" onClick={() => setLightbox(null)}>
          <img src={lightbox} alt="preview" />
        </div>
      )}
    </div>
  );
}
