import Link from "next/link";

export default function Home() {
  return (
    <main className="landing-page">
      <section className="landing-card" aria-labelledby="aura-title">
        <div className="landing-mark" aria-hidden="true">A</div>
        <p className="landing-kicker">Artificial Understanding of Research Articles</p>
        <h1 id="aura-title">Hi, I&apos;m AURA.</h1>
        <p className="landing-copy">New to machine learning? Research papers can feel like a new language. Ask AURA what a method means, work through the maths, and keep asking until it clicks.</p>
        <p className="landing-credit">Open source, built by Antriksh Arya.</p>
        <div className="landing-actions">
          <a className="landing-link" href="https://github.com/vdhkcheems/AURA" target="_blank" rel="noreferrer">Contribute / repo <span>↗</span></a>
          <Link className="landing-start" href="/chat">Start chatting <span>→</span></Link>
        </div>
      </section>
    </main>
  );
}
