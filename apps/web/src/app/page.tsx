export default function Home() {
  return (
    <main className="shell">
      <p className="eyebrow">AURA · research paper Q&amp;A</p>
      <h1>Paper-grounded answers are ready.</h1>
      <p className="lede">
        The server-side retrieval API can search the indexed machine-learning corpus and return answer-ready sources.
      </p>
      <section className="endpoint-card" aria-labelledby="api-title">
        <h2 id="api-title">API foundation</h2>
        <code>POST /api/chat</code>
        <p>Send a question, optionally scoped to a paper or topic. The interactive chat interface is the next layer.</p>
        <code>GET /api/health</code>
        <p>Checks server configuration and Qdrant index availability.</p>
      </section>
    </main>
  );
}
