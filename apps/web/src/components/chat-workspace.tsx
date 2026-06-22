"use client";

import { FormEvent, KeyboardEvent, useEffect, useMemo, useState } from "react";
import ReactMarkdown from "react-markdown";
import rehypeKatex from "rehype-katex";
import remarkGfm from "remark-gfm";
import remarkMath from "remark-math";

import { supportedPapers } from "@/lib/papers";

type Source = {
  chunkId: string;
  paperId: string;
  title: string;
  sectionPath: string[];
  text: string;
  sourceFiles: string[];
  score: number;
};

type ChatStreamEvent =
  | { type: "meta"; sources: Source[] }
  | { type: "delta"; text: string }
  | { type: "done" }
  | { type: "error"; error: string };

type Message = {
  id: string;
  role: "user" | "assistant";
  content: string;
  sources?: Source[];
  pending?: boolean;
  error?: boolean;
};

type Chat = {
  id: string;
  title: string;
  createdAt: string;
  updatedAt: string;
  paperId?: string;
  messages: Message[];
};

const storageKey = "aura.guest-chats.v1";
const themeStorageKey = "aura.theme.v1";
const suggestedQuestions = [
  "What problem does this paper solve?",
  "Explain the core idea simply.",
  "What equation should I understand first?",
];

function newChat(paperId?: string): Chat {
  const now = new Date().toISOString();
  return { id: crypto.randomUUID(), title: "New conversation", createdAt: now, updatedAt: now, paperId, messages: [] };
}

function titleFor(question: string) {
  return question.length > 42 ? `${question.slice(0, 42).trim()}…` : question;
}

function readChats(): Chat[] {
  try {
    const value = window.localStorage.getItem(storageKey);
    const chats = value ? JSON.parse(value) : [];
    return Array.isArray(chats) ? chats.filter(isChat) : [];
  } catch {
    return [];
  }
}

function isChat(value: unknown): value is Chat {
  return Boolean(value && typeof value === "object" && "id" in value && "messages" in value);
}

export function ChatWorkspace() {
  const [chats, setChats] = useState<Chat[]>([]);
  const [activeId, setActiveId] = useState<string>("");
  const [input, setInput] = useState("");
  const [isReady, setIsReady] = useState(false);
  const [accountNotice, setAccountNotice] = useState(false);
  const [expandedSources, setExpandedSources] = useState<string | null>(null);
  const [theme, setTheme] = useState<"light" | "dark">("light");

  useEffect(() => {
    const saved = readChats();
    const initial = saved.length ? saved : [newChat()];
    // This is client-only hydration from the browser's persistent guest store.
    // eslint-disable-next-line react-hooks/set-state-in-effect
    setChats(initial);
    setActiveId(initial[0].id);
    setIsReady(true);
  }, []);

  useEffect(() => {
    // Theme preference exists only after client hydration.
    // eslint-disable-next-line react-hooks/set-state-in-effect
    if (window.localStorage.getItem(themeStorageKey) === "dark") setTheme("dark");
  }, []);

  useEffect(() => {
    if (isReady) window.localStorage.setItem(themeStorageKey, theme);
  }, [isReady, theme]);

  useEffect(() => {
    if (isReady) window.localStorage.setItem(storageKey, JSON.stringify(chats));
  }, [chats, isReady]);

  const activeChat = useMemo(
    () => chats.find((chat) => chat.id === activeId) ?? chats[0],
    [activeId, chats],
  );
  const activePaper = supportedPapers.find((paper) => paper.id === activeChat?.paperId);

  function updateChat(chatId: string, transform: (chat: Chat) => Chat) {
    setChats((current) => current.map((chat) => (chat.id === chatId ? transform(chat) : chat)));
  }

  function createChat(paperId?: string) {
    const chat = newChat(paperId);
    setChats((current) => [chat, ...current]);
    setActiveId(chat.id);
    setInput("");
  }

  function deleteChat(chatId: string) {
    const remaining = chats.filter((chat) => chat.id !== chatId);
    const next = remaining.length ? remaining : [newChat()];
    setChats(next);
    if (activeId === chatId) setActiveId(next[0].id);
  }

  function renameChat(chat: Chat) {
    const next = window.prompt("Name this conversation", chat.title)?.trim();
    if (next) updateChat(chat.id, (current) => ({ ...current, title: next, updatedAt: new Date().toISOString() }));
  }

  async function sendMessage(event?: FormEvent, suggested?: string) {
    event?.preventDefault();
    const question = (suggested ?? input).trim();
    if (!question || !activeChat || activeChat.messages.some((message) => message.pending)) return;
    const pendingId = crypto.randomUUID();
    const userMessage: Message = { id: crypto.randomUUID(), role: "user", content: question };
    const history = activeChat.messages
      .filter((message) => !message.pending && !message.error)
      .slice(-12)
      .map((message) => ({ role: message.role, content: message.content }));
    const now = new Date().toISOString();
    updateChat(activeChat.id, (chat) => ({
      ...chat,
      title: chat.messages.length ? chat.title : titleFor(question),
      updatedAt: now,
      messages: [...chat.messages, userMessage, { id: pendingId, role: "assistant", content: "", pending: true }],
    }));
    setInput("");

    try {
      const response = await fetch("/api/chat", {
        method: "POST",
        headers: { "content-type": "application/json" },
        body: JSON.stringify({ question, paperId: activeChat.paperId, history }),
      });
      if (!response.ok || !response.body) {
        const data = await response.json().catch(() => ({})) as { error?: string };
        throw new Error(data.error || "AURA could not answer that right now.");
      }
      const reader = response.body.getReader();
      const decoder = new TextDecoder();
      let buffer = "";
      let answer = "";
      let sources: Source[] = [];
      let completed = false;
      const handleEvent = (event: ChatStreamEvent) => {
        if (event.type === "meta") sources = event.sources;
        if (event.type === "delta") {
          answer += event.text;
          updateChat(activeChat.id, (chat) => ({ ...chat, messages: chat.messages.map((message) => message.id === pendingId ? { ...message, content: answer } : message) }));
        }
        if (event.type === "error") throw new Error(event.error);
        if (event.type === "done") completed = true;
      };
      while (true) {
        const { done, value } = await reader.read();
        buffer += decoder.decode(value, { stream: !done });
        const lines = buffer.split("\n");
        buffer = lines.pop() ?? "";
        for (const line of lines) if (line) handleEvent(JSON.parse(line) as ChatStreamEvent);
        if (done) break;
      }
      if (buffer) handleEvent(JSON.parse(buffer) as ChatStreamEvent);
      if (!completed || !answer) throw new Error("AURA did not complete the answer. Please try again.");
      updateChat(activeChat.id, (chat) => ({ ...chat, updatedAt: new Date().toISOString(), messages: chat.messages.map((message) => message.id === pendingId ? { ...message, pending: false, sources } : message) }));
    } catch (error) {
      const message = error instanceof Error ? error.message : "AURA could not answer that right now.";
      updateChat(activeChat.id, (chat) => ({
        ...chat,
        messages: chat.messages.map((item) => item.id === pendingId
          ? { id: pendingId, role: "assistant", content: message, error: true }
          : item),
      }));
    }
  }

  function handleComposerKeyDown(event: KeyboardEvent<HTMLTextAreaElement>) {
    if (event.key !== "Enter" || event.shiftKey || event.nativeEvent.isComposing) return;
    event.preventDefault();
    void sendMessage();
  }

  if (!isReady || !activeChat) return <main className="loading-shell">Opening your research desk…</main>;

  return (
    <main className="app-shell" data-theme={theme}>
      <aside className="sidebar">
        <div className="brand-row"><div className="brand-mark">A</div><span>AURA</span></div>
        <button className="new-chat" onClick={() => createChat()}>＋ New chat</button>
        <div className="sidebar-label">Your guest chats</div>
        <nav className="chat-list" aria-label="Guest chat history">
          {chats.map((chat) => (
            <div className={`chat-row ${chat.id === activeChat.id ? "active" : ""}`} key={chat.id}>
              <button onClick={() => setActiveId(chat.id)}>{chat.title}</button>
              <button aria-label={`Delete ${chat.title}`} className="delete-chat" onClick={() => deleteChat(chat.id)}>×</button>
            </div>
          ))}
        </nav>
        <div className="library">
          <div className="sidebar-label">Supported papers <span>{supportedPapers.length}</span></div>
          {supportedPapers.map((paper) => (
            <button className={`paper-card ${paper.id === activeChat.paperId ? "selected" : ""}`} key={paper.id} onClick={() => createChat(paper.id)}>
              <strong>{paper.title}</strong><small>{paper.year} · {paper.topics.join(" · ")}</small>
            </button>
          ))}
        </div>
        <div className="guest-card"><strong>Guest workspace</strong><span>Chats stay in this browser.</span><button onClick={() => setAccountNotice(true)}>Sign in to sync <span>↗</span></button></div>
      </aside>

      <section className="conversation">
        <header className="conversation-header">
          <div><p className="eyebrow">{activePaper ? "Paper conversation" : "Research library"}</p><h1>{activePaper?.title ?? "Talk to machine-learning papers"}</h1></div>
          <div className="header-actions"><button className="theme-toggle" type="button" onClick={() => setTheme((current) => current === "light" ? "dark" : "light")} aria-label={theme === "light" ? "Switch to dark mode" : "Switch to light mode"} title={theme === "light" ? "Switch to dark mode" : "Switch to light mode"}>{theme === "light" ? "◐" : "☀"}</button><button className="rename" onClick={() => renameChat(activeChat)}>Rename</button></div>
        </header>

        <div className="scope-bar">
          <span>Searching</span>
          <select value={activeChat.paperId ?? ""} onChange={(event) => updateChat(activeChat.id, (chat) => ({ ...chat, paperId: event.target.value || undefined }))}>
            <option value="">All supported papers</option>
            {supportedPapers.map((paper) => <option value={paper.id} key={paper.id}>{paper.title}</option>)}
          </select>
        </div>

        <div className="messages" aria-live="polite">
          {activeChat.messages.length === 0 ? (
            <div className="welcome"><p className="welcome-kicker">Research papers, in conversation.</p><h2>Start with what feels confusing.</h2><p>AURA retrieves evidence from the papers before answering, so you can follow up without losing the thread.</p><div className="suggestions">{suggestedQuestions.map((question) => <button onClick={() => sendMessage(undefined, question)} key={question}>{question}</button>)}</div></div>
          ) : activeChat.messages.map((message) => <MessageView key={message.id} message={message} expandedSources={expandedSources} setExpandedSources={setExpandedSources} />)}
        </div>

        <form className="composer" onSubmit={sendMessage}>
          <textarea value={input} onChange={(event) => setInput(event.target.value)} onKeyDown={handleComposerKeyDown} placeholder="Ask about a concept, equation, experiment, or comparison…" rows={2} />
          <div><span>Guest chats are saved in this browser.</span><button type="submit" disabled={!input.trim() || activeChat.messages.some((message) => message.pending)}>Ask AURA <span>↑</span></button></div>
        </form>
      </section>

      {accountNotice && <div className="modal-backdrop" role="presentation" onClick={() => setAccountNotice(false)}><section className="account-modal" role="dialog" aria-modal="true" aria-label="Account sync coming soon" onClick={(event) => event.stopPropagation()}><button className="modal-close" onClick={() => setAccountNotice(false)}>×</button><p className="eyebrow">Coming soon</p><h2>Take your research desk with you.</h2><p>Accounts will let you sync conversations across devices and optionally import your guest chats. For now, your conversations remain private to this browser.</p><button onClick={() => setAccountNotice(false)}>Continue as guest</button></section></div>}
    </main>
  );
}

function MessageView({ message, expandedSources, setExpandedSources }: { message: Message; expandedSources: string | null; setExpandedSources: (value: string | null) => void }) {
  if (message.pending && !message.content) return <article className="message assistant pending"><div className="message-label">AURA is reading</div><div className="thinking"><i /><i /><i /></div></article>;
  if (message.role === "user") return <article className="message user"><div className="message-label">You</div><p>{message.content}</p></article>;
  return <article className={`message assistant ${message.error ? "error" : ""}`}><div className="message-label">AURA</div><div className="answer-markdown"><ReactMarkdown remarkPlugins={[remarkGfm, remarkMath]} rehypePlugins={[rehypeKatex]}>{message.content}</ReactMarkdown></div>{message.sources && message.sources.length > 0 && <div className="sources"><button className="source-toggle" onClick={() => setExpandedSources(expandedSources === message.id ? null : message.id)}>Sources used <span>{message.sources.length}</span><b>{expandedSources === message.id ? "−" : "+"}</b></button>{expandedSources === message.id && <div className="source-list">{message.sources.map((source, index) => <article className="source" key={source.chunkId}><div><span>[{index + 1}]</span><strong>{source.title}</strong><small>{source.sectionPath.join(" › ")}</small></div><p>{source.text}</p></article>)}</div>}</div>}</article>;
}
