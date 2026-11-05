import { useState } from 'react'
import { BookOpen, Lightbulb, Search, ShieldCheck, Sparkles } from 'lucide-react'
import { apiJson } from '../services/api'
import './idea-checker.css'

export default function IdeaCheckerPage() {
  const [title, setTitle] = useState('')
  const [abstract, setAbstract] = useState('')
  const [result, setResult] = useState(null)
  const [error, setError] = useState('')
  const [busy, setBusy] = useState(false)

  async function submit(event) {
    event.preventDefault(); setBusy(true); setError(''); setResult(null)
    try {
      const response = await apiJson('/ideas/check', { method: 'POST', body: JSON.stringify({ title, abstract }) })
      setResult(response.result)
    } catch (requestError) { setError(requestError.message) } finally { setBusy(false) }
  }

  return <div className="page-stack idea-page">
    <div className="page-intro"><div><span className="section-kicker">BEFORE YOU SUBMIT</span><h1>Check a project idea</h1><p>Compare a proposed title and abstract with eligible projects in the institutional repository.</p></div></div>
    <div className="idea-layout">
      <form className="panel idea-form" onSubmit={submit}>
        <div className="idea-heading"><span className="idea-icon"><Lightbulb size={19} /></span><div><h2>Proposed project</h2><p>Enter the idea before uploading a full document.</p></div></div>
        <label>Project title<input required minLength="3" maxLength="140" placeholder="A specific, working title" value={title} onChange={(event) => setTitle(event.target.value)} /></label>
        <label>Abstract<textarea required minLength="30" maxLength="1200" rows="7" placeholder="Describe the problem, intended approach, and expected contribution…" value={abstract} onChange={(event) => setAbstract(event.target.value)} /><small>{abstract.length}/1200</small></label>
        {error && <p className="idea-error">{error}</p>}
        <button className="btn btn-primary" disabled={busy}>{busy ? <><span className="spinner" /> Comparing idea…</> : <>Check against repository <Search size={16} /></>}</button>
        <p className="idea-privacy"><ShieldCheck size={14} /> This check does not create or upload a project record.</p>
      </form>
      <aside className="idea-method"><span className="section-kicker">COMPARISON METHOD</span><h2>Use the signal to refine the question.</h2><p>Title and abstract are compared with eligible project records using TF-IDF and Sentence-BERT when available.</p><div><BookOpen size={16} /><span>Every match is an existing repository record.</span></div><div><ShieldCheck size={16} /><span>Similarity is not a plagiarism decision.</span></div></aside>
    </div>
    {result && <section className="panel idea-results"><div className="idea-result-head"><div><span className="section-kicker">IDEA CHECK · {result.similar_projects.length} COMPARISONS</span><h2>Closest existing ideas</h2></div><span className={`idea-method-pill ${result.semantic_available ? 'model-on' : ''}`}><Sparkles size={14} />{result.semantic_available ? 'Sentence-BERT enabled' : 'Semantic analysis unavailable · TF-IDF fallback'}</span></div><div className="idea-score-row"><div><small>TEXTUAL</small><b>{Number(result.textual_similarity).toFixed(1)}%</b></div><div><small>SEMANTIC</small><b>{result.semantic_available ? `${Number(result.semantic_similarity).toFixed(1)}%` : 'Unavailable'}</b></div><div><small>OVERALL</small><b>{Number(result.overall_similarity).toFixed(1)}%</b></div></div><div className="idea-match-list">{result.similar_projects.map((item, index) => <article key={item.reference.id} className="idea-match"><span className="idea-rank">{String(index + 1).padStart(2, '0')}</span><div className="idea-match-copy"><div><h3>{item.reference.title}</h3>{item.reference.is_sample && <span className="demo-chip">DEMO</span>}</div><p>{item.reference.abstract}</p><small>{item.reference.department} · {item.reference.academic_year}</small></div><div className="idea-match-scores"><b>{Number(item.overall_similarity).toFixed(1)}%</b><small>{Number(item.textual_similarity).toFixed(1)}% text</small><small>{item.semantic_available ? `${Number(item.semantic_similarity).toFixed(1)}% semantic` : 'semantic unavailable'}</small></div></article>)}</div><div className="idea-suggestions"><div><Lightbulb size={16} /><b>Originality suggestions · not guarantees</b></div><ul>{result.suggestions.map((suggestion) => <li key={suggestion}>{suggestion}</li>)}</ul></div><p className="idea-disclaimer">Similarity scores indicate textual or semantic overlap and do not independently establish plagiarism. Final academic decisions require faculty review.</p></section>}
  </div>
}
