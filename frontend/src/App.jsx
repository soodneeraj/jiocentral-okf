import React, { useCallback, useEffect, useMemo, useRef, useState } from "react";

const API = "";

const PROGRESS_STEPS = [
  "Understanding your question…",
  "Searching document text…",
  "Searching figures…",
  "Assembling evidence…",
  "Generating grounded answer…",
];

function imageUrl(path) {
  if (!path) return "";
  if (path.startsWith("http") || path.startsWith("/")) return path;
  const name = path.split("/").pop();
  if (name && /\.(png|jpe?g|webp|gif|bmp|svg)$/i.test(name)) {
    return `/static/images/${encodeURIComponent(name)}`;
  }
  return `/static/bundle/${path.split("/").map(encodeURIComponent).join("/")}`;
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

function renderInline(text) {
  if (!text) return null;
  // Minimal markdown: **bold**, *italic*, newlines
  const escaped = text
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;");
  const withMd = escaped
    .replace(/\*\*\*(.+?)\*\*\*/g, "<strong><em>$1</em></strong>")
    .replace(/\*\*(.+?)\*\*/g, "<strong>$1</strong>")
    .replace(/\*([^*\n]+?)\*/g, "<em>$1</em>")
    .replace(/\n/g, "<br/>");
  return <span dangerouslySetInnerHTML={{ __html: withMd }} />;
}

export default function App() {
  const [conversations, setConversations] = useState([]);
  const [messages, setMessages] = useState([]);
  const [input, setInput] = useState("");
  const [busy, setBusy] = useState(false);
  const [progressElapsedMs, setProgressElapsedMs] = useState(0);
  const [progressStep, setProgressStep] = useState(0);
  const [conversationId, setConversationId] = useState(null);
  const [lightbox, setLightbox] = useState(null);
  const [docHtmlParts, setDocHtmlParts] = useState(null);
  const [sidebarOpen, setSidebarOpen] = useState(true);
  const messagesEndRef = useRef(null);
  const tickRef = useRef(null);
  const stepRef = useRef(null);

  const refreshConversations = useCallback(async () => {
    try {
      const res = await fetch(`${API}/api/v1/chat/conversations`);
      const data = await res.json();
      setConversations(data.conversations || []);
    } catch (e) {
      console.error(e);
    }
  }, []);

  useEffect(() => {
    refreshConversations();
  }, [refreshConversations]);

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

  function newChat() {
    clearProgressTimers();
    setMessages([]);
    setConversationId(null);
    setDocHtmlParts(null);
    setBusy(false);
    setProgressElapsedMs(0);
    setInput("");
  }

  async function openConversation(id) {
    if (!id || busy) return;
    try {
      const res = await fetch(`${API}/api/v1/chat/conversations/${id}`);
      if (!res.ok) return;
      const data = await res.json();
      setConversationId(data.conversation_id);
      const mapped = (data.messages || []).map((m) => {
        const evidence = m.evidence || {};
        return {
          role: m.role,
          content: m.content || "",
          sources: evidence.sources || [],
          images: evidence.images || [],
        };
      });
      setMessages(mapped);
      const lastAi = [...mapped].reverse().find((m) => m.role === "assistant");
      setDocHtmlParts(lastAi || null);
    } catch (e) {
      console.error(e);
    }
  }

  async function deleteConversation(id, e) {
    e?.stopPropagation?.();
    if (!id) return;
    if (!window.confirm("Delete this chat?")) return;
    try {
      await fetch(`${API}/api/v1/chat/conversations/${id}`, { method: "DELETE" });
      if (conversationId === id) newChat();
      await refreshConversations();
    } catch (err) {
      console.error(err);
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
      const assistantMsg = {
        role: "assistant",
        content: data.answer || "",
        sources: data.sources || [],
        images: data.images || [],
        warnings: data.warnings || [],
        elapsedMs,
      };
      setMessages((m) => [...m, assistantMsg]);
      setDocHtmlParts(assistantMsg);
      await refreshConversations();
    } catch (err) {
      const elapsedMs = performance.now() - t0;
      setMessages((m) => [
        ...m,
        { role: "assistant", content: `Error: ${err.message}`, elapsedMs },
      ]);
    } finally {
      clearProgressTimers();
      setBusy(false);
      setProgressElapsedMs(0);
      setProgressStep(0);
    }
  }

  const latestImages = useMemo(() => {
    if (docHtmlParts?.images?.length) return docHtmlParts.images;
    for (let i = messages.length - 1; i >= 0; i--) {
      if (messages[i].images?.length) return messages[i].images;
    }
    return [];
  }, [messages, docHtmlParts]);

  return (
    <div className={`app ${sidebarOpen ? "" : "sidebar-collapsed"}`}>
      <aside className="sidebar">
        <div className="side-top">
          <div className="brand">OKF Chat</div>
          <button className="icon-btn" onClick={() => setSidebarOpen(false)} title="Collapse">
            ‹
          </button>
        </div>
        <button className="new-chat" onClick={newChat}>
          + New chat
        </button>
        <div className="hist-label">Chat history</div>
        <div className="hist-list">
          {conversations.length === 0 && (
            <div className="hist-empty">No chats yet</div>
          )}
          {conversations.map((c) => (
            <div
              key={c.conversation_id}
              className={`hist-item ${
                c.conversation_id === conversationId ? "active" : ""
              }`}
              onClick={() => openConversation(c.conversation_id)}
            >
              <span className="hist-title">{c.title || "Untitled"}</span>
              <button
                className="hist-del"
                title="Delete"
                onClick={(e) => deleteConversation(c.conversation_id, e)}
              >
                ×
              </button>
            </div>
          ))}
        </div>
        <div className="side-foot">
          <a href="/admin" target="_blank" rel="noopener">
            Admin
          </a>
        </div>
      </aside>

      {!sidebarOpen && (
        <button className="reopen" onClick={() => setSidebarOpen(true)} title="History">
          ☰
        </button>
      )}

      <main className="main">
        <div className="messages">
          {messages.length === 0 && !busy && (
            <div className="hero">
              <h1>How can I help?</h1>
              <p>Ask about procedures, steps, and figures in your approved documents.</p>
            </div>
          )}
          {messages.map((m, idx) => (
            <React.Fragment key={idx}>
              <div className={`row ${m.role}`}>
                <div className={`bubble ${m.role === "user" ? "user" : "ai"}`}>
                  {renderInline(m.content)}
                  {m.role === "assistant" && m.elapsedMs != null && (
                    <div className="meta-time">
                      {formatDuration(m.elapsedMs)}
                    </div>
                  )}
                  {m.warnings?.length > 0 && (
                    <div className="warn">Warnings: {m.warnings.join(", ")}</div>
                  )}
                </div>
              </div>
              {m.role === "assistant" &&
                (m.sources?.length > 0 || m.images?.length > 0) && (
                  <div className="row assistant">
                    <div className="bubble evidence">
                      <strong>Sources</strong>
                      {(m.sources || []).map((s, i) => (
                        <div key={i}>
                          {s.source} {s.source_loc}
                          {s.page != null ? ` (page ${s.page})` : ""}
                        </div>
                      ))}
                      {(m.images || []).length > 0 && (
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
                      )}
                    </div>
                  </div>
                )}
            </React.Fragment>
          ))}

          {busy && (
            <div className="row assistant">
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
              </div>
            </div>
          )}
          <div ref={messagesEndRef} />
        </div>

        <div className="composer-wrap">
          <div className="composer">
            <textarea
              value={input}
              placeholder={busy ? "Waiting for response…" : "Message OKF…"}
              disabled={busy}
              rows={1}
              onChange={(e) => setInput(e.target.value)}
              onKeyDown={(e) => {
                if (e.key === "Enter" && !e.shiftKey) {
                  e.preventDefault();
                  send();
                }
              }}
            />
            <button onClick={send} disabled={busy || !input.trim()}>
              {busy ? "…" : "Send"}
            </button>
          </div>
          <div className="composer-hint">Enter to send · Shift+Enter for newline</div>
        </div>
      </main>

      <aside className="rail">
        <div className="rail-title">Document view</div>
        <div className="rail-body">
          {!docHtmlParts && (
            <p className="muted">Formatted answer content and figures appear here.</p>
          )}
          {docHtmlParts && (
            <>
              <div className="rail-answer">{renderInline(docHtmlParts.content)}</div>
              <div className="thumbs">
                {latestImages.map((img, i) => (
                  <div key={i}>
                    <img
                      src={imageUrl(img.image_path)}
                      alt=""
                      onClick={() => setLightbox(imageUrl(img.image_path))}
                    />
                    <div className="muted">p{img.page}</div>
                  </div>
                ))}
              </div>
            </>
          )}
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
