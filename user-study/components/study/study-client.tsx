'use client';
import { useCallback, useEffect, useRef, useState } from 'react';
import { Check, CheckCircle2, ChevronLeft, Expand, FileText, FlaskConical, Layers3, Save, ShieldCheck, X, ZoomIn, ZoomOut } from 'lucide-react';
import { METRICS, isComplete, type PublicDiagram, type StudySession, type Rating } from '../../lib/metrics';

async function api(path: string, method = 'GET', body?: unknown) {
  const r = await fetch(path, {method, headers: {'Content-Type':'application/json'}, ...(body ? {body:JSON.stringify(body)} : {})});
  const data = await r.json() as {error?: string; session: StudySession; rating: Rating};
  if (!r.ok) throw new Error(data.error || 'We could not save your work. Please try again.');
  return data;
}
export function Brand() { return <a href="/" className="brand"><span className="brand-icon"><Layers3 size={21}/></span><span>Diagram<span className="brand-light">Lab</span></span><span className="brand-divider"/><span className="brand-caption">Human evaluation</span></a>; }
export default function StudyClient({preview,sampleSize,demo}:{preview:PublicDiagram;sampleSize:number;demo:boolean}) {
  const [session,setSession]=useState<StudySession|null>(null), [loading,setLoading]=useState(true), [index,setIndex]=useState(0);
  const [group,setGroup]=useState<'correctness'|'beauty'>('correctness');
  const [draft,setDraft]=useState<Rating>({scores:{},comment:'',durationMs:0});
  const [status,setStatus]=useState(''), [error,setError]=useState(''), [busy,setBusy]=useState(false), [consent,setConsent]=useState(false), [name,setName]=useState('');
  const [zoom,setZoom]=useState(1), [expanded,setExpanded]=useState(false), [review,setReview]=useState(false);
  const elapsed=useRef(0), activeAt=useRef(0), draftRef=useRef(draft), revision=useRef(0), queue=useRef(Promise.resolve()), dirty=useRef(false);
  const diagram=session?.diagrams[index] || preview;
  const sessionId=session?.id, locked=!!session?.submittedAt;
  const completed=session ? session.diagrams.filter(d=>isComplete(session.ratings[d.id]?.scores||{})).length : 0;
  const count=Object.keys(draft.scores).length;
  const total=session?.diagrams.length || sampleSize;
  const loadDraft=useCallback((s:StudySession,i:number)=>{
    const saved=s.ratings[s.diagrams[i].id] || {scores:{},comment:'',durationMs:0};
    setDraft(saved);draftRef.current=saved;elapsed.current=saved.durationMs;activeAt.current=Date.now();dirty.current=false;revision.current++;
    setGroup('correctness');setZoom(1);setStatus(saved.updatedAt?'Saved':'');
  },[]);
  useEffect(()=>{api('/api/session').then(data=>{if(data.session){const s=data.session as StudySession;setSession(s);const i=Math.max(0,s.diagrams.findIndex(d=>!isComplete(s.ratings[d.id]?.scores||{})));setIndex(i);loadDraft(s,i);}}).catch(e=>setError(e.message)).finally(()=>setLoading(false));},[loadDraft]);
  useEffect(()=>{
    activeAt.current=Date.now();
    const tick=()=>{const now=Date.now();if(document.visibilityState==='visible') elapsed.current+=Math.min(now-activeAt.current,2000);activeAt.current=now;};
    const timer=setInterval(tick,1000);
    const reset=()=>{activeAt.current=Date.now();};document.addEventListener('visibilitychange',reset);
    return()=>{clearInterval(timer);document.removeEventListener('visibilitychange',reset);};
  },[]);
  const persist=useCallback(async()=>{
    if(!sessionId || locked) return;
    const saved={...draftRef.current,durationMs:Math.round(elapsed.current)}, id=diagram.id, version=revision.current;
    setStatus('Saving…');
    const job=queue.current.catch(()=>{}).then(async()=>{
      const data=await api('/api/rating','PUT',{diagramId:id,...saved});
      setSession(current=>current?{...current,ratings:{...current.ratings,[id]:data.rating}}:current);
      if(version===revision.current){dirty.current=false;setStatus('All changes saved');}
    });
    queue.current=job;
    try { await job;setError(''); } catch(e){setStatus('Not saved');setError((e as Error).message);throw e;}
  },[sessionId,locked,diagram.id]);
  useEffect(()=>{if(!dirty.current || !sessionId || locked) return;const timer=setTimeout(()=>{void persist().catch(()=>{});},650);return()=>clearTimeout(timer);},[draft,persist,sessionId,locked]);
  useEffect(()=>{const warn=(e:BeforeUnloadEvent)=>{if(dirty.current){e.preventDefault();e.returnValue='';}};window.addEventListener('beforeunload',warn);return()=>window.removeEventListener('beforeunload',warn);},[]);
  useEffect(()=>{if(!expanded)return;const esc=(e:KeyboardEvent)=>{if(e.key==='Escape')setExpanded(false);};window.addEventListener('keydown',esc);return()=>window.removeEventListener('keydown',esc);},[expanded]);
  function edit(next:Rating){revision.current++;dirty.current=true;draftRef.current=next;setDraft(next);setStatus('Unsaved changes');}
  async function start(){setBusy(true);setError('');try{const d=await api('/api/session','POST',{consent:true,name});setSession(d.session);setIndex(0);loadDraft(d.session,0);}catch(e){setError((e as Error).message);}finally{setBusy(false);}}
  async function navigate(next:number){setBusy(true);try{await persist();setIndex(next);const savedSession={...session!,ratings:{...session!.ratings,[diagram.id]:{...draftRef.current,durationMs:elapsed.current}}};loadDraft(savedSession,next);setReview(false);}catch{}finally{setBusy(false);}}
  async function next(){if(!isComplete(draftRef.current.scores))return;if(index<total-1){await navigate(index+1);}else{setBusy(true);try{await persist();setReview(true);}catch{}finally{setBusy(false);}}}
  async function submit(){setBusy(true);try{await persist();const data=await api('/api/submit','POST',{});setSession(data.session);setReview(false);}catch(e){setError((e as Error).message);}finally{setBusy(false);}}
  const progressRef=useRef({completed:0,total:sampleSize,current:1,submitted:false,missing:[] as string[]});
  useEffect(()=>{progressRef.current={completed,total,current:index+1,submitted:locked,missing:METRICS.filter(m=>!draft.scores[m.id]).map(m=>m.name)};});
  useEffect(()=>{
    const context=(document as Document & {modelContext?:{registerTool:(t:unknown,o:unknown)=>unknown}}).modelContext;
    if(!context)return;const lifecycle=new AbortController();
    try{void Promise.resolve(context.registerTool({name:'read_study_progress',description:'Read study progress and missing rating names. This does not create or submit human ratings.',inputSchema:{type:'object',properties:{},additionalProperties:false},annotations:{readOnlyHint:true},execute:(input:unknown)=>{if(!input||typeof input!=='object'||Array.isArray(input)||Object.keys(input).length)throw new Error('Expected an empty object.');return progressRef.current;}},{signal:lifecycle.signal})).catch(()=>{});}catch{}
    return()=>lifecycle.abort();
  },[]);
  const currentComplete=isComplete(draft.scores);
  return <>
    <header className="topbar"><Brand/><div className="topbar-right">{demo&&<span className="demo-pill"><FlaskConical size={14}/> Demo study</span>}<a className="quiet-link" href="/researcher">Researcher workspace</a></div></header>
    {session?.submittedAt ? <main className="finish-wrap"><div className="finish-icon"><Check size={32}/></div><p className="eyebrow">STUDY COMPLETE</p><h1>Thank you for your perspective.</h1><p>Your ratings for all {total} diagrams have been saved. You can safely close this page.</p><div className="receipt"><span>Completion code</span><strong>{session.participantCode}</strong><small>{demo?'This was a demo session.':'Keep this code for your records.'}</small></div><a href="/researcher" className="quiet-link">Researcher workspace</a></main> : <main className="study-shell">
      <section className="study-heading"><div><p className="eyebrow">DIAGRAM EVALUATION STUDY</p><h1>Every detail deserves a closer look.</h1><p>Review each diagram and its caption. Rate each dimension from 1 to 10.</p></div><div className="progress-summary"><span><strong>{completed}</strong> / {total} diagrams</span><span className="progress-track"><i style={{width:`${completed/total*100}%`}}/></span><small>{session?`${session.participantName||session.participantCode} · Progress saved on this browser`:'Your responses are saved as you go'}</small></div></section>
      {error&&<div className="error-banner" role="alert">{error}<button onClick={()=>{void persist().catch(()=>{});}}>Retry saving</button></div>}
      <div className="study-grid">
        <section className="visual-column">
          <div className="diagram-card"><div className="card-toolbar"><span><i className="blue-dot"/> {session?`Diagram ${String(index+1).padStart(2,'0')}`:'Preview diagram'}<span className="muted-inline">{session?`of ${total}`:'Explore before you begin'}</span></span><div className="icon-tools"><button title="Zoom out" aria-label="Zoom out" onClick={()=>setZoom(z=>Math.max(.5,z-.25))}><ZoomOut size={17}/></button><button className="zoom-reset" title="Reset zoom" onClick={()=>setZoom(1)}>{Math.round(zoom*100)}%</button><button title="Zoom in" aria-label="Zoom in" onClick={()=>setZoom(z=>Math.min(3,z+.25))}><ZoomIn size={17}/></button><span/><button title="Expand diagram" aria-label="Expand diagram" onClick={()=>setExpanded(true)}><Expand size={17}/></button></div></div><div className="diagram-canvas"><img src={diagram.image} alt="Diagram to evaluate against its caption below" style={{width:`${zoom*100}%`,height:`${zoom*100}%`,maxWidth:'none'}}/></div><div className="canvas-foot"><span>Inspect the whole diagram, including labels and arrows.</span><span>Zoom to inspect</span></div></div>
          <div className="description-card"><div className="section-label"><FileText size={17}/><h2>Caption</h2></div><div className="description-text">{diagram.caption}</div></div>
          <div className="session-note"><ShieldCheck size={17}/><span>Evaluate independently; there are no suggested scores.</span></div>
          {session&&<div className="diagram-navigation"><span className="nav-caption">YOUR DIAGRAMS</span><div className="diagram-dots">{session.diagrams.map((d,i)=><button key={d.id} disabled={busy} aria-label={`Go to diagram ${i+1}${isComplete(session.ratings[d.id]?.scores||{})?', rated':''}`} aria-current={i===index?'step':undefined} className={`${i===index?'active ':''}${isComplete(session.ratings[d.id]?.scores||{})?'rated':''}`} onClick={()=>{void navigate(i);}}>{isComplete(session.ratings[d.id]?.scores||{})?<Check size={13}/>:String(i+1).padStart(2,'0')}</button>)}</div></div>}
        </section>
        <section className="rating-column">
          {!session&&<div className="start-panel"><div className="start-title"><span className="step-circle">01</span><div><h2>Ready to take a closer look?</h2><p>{total} diagrams · 10 dimensions each · 1–10 scale</p></div></div><label className="name-field" htmlFor="participant-name">Your name</label><input className="text-input" id="participant-name" autoComplete="name" maxLength={80} value={name} onChange={e=>setName(e.target.value)} placeholder="First and last name"/><label className="consent"><input type="checkbox" checked={consent} onChange={e=>setConsent(e.target.checked)}/><span>I agree to have my name and ratings stored for this {demo?'demo':'research'} study. I can stop at any time.</span></label><button className="primary full" onClick={start} disabled={!consent||!name.trim()||busy||loading}>{loading?'Connecting…':busy?'Starting…':'Begin rating'}</button>{demo&&<small>Demo images and provisional difficulty labels. Responses are marked as demo data.</small>}</div>}
          <div className="ratings-panel"><div className="rating-title"><div><p className="eyebrow">YOUR ASSESSMENT</p><h2>{group==='correctness'?'Does it communicate correctly?':'How does it look and feel?'}</h2></div><span className="count-tag">{count}<span>/10</span></span></div><div className="group-tabs" role="tablist" aria-label="Rating categories">{(['correctness','beauty'] as const).map(g=><button role="tab" aria-selected={group===g} key={g} className={group===g?'selected':''} onClick={()=>setGroup(g)}>{g==='correctness'?'Correctness':'Beauty'}<span>{METRICS.filter(m=>m.group===g&&draft.scores[m.id]).length}/5</span></button>)}</div><div className="scale-guide"><span><b>1</b> Very poor</span><span>Choose one score per dimension</span><span><b>10</b> Excellent</span></div>
            <div className="metric-list">{METRICS.filter(m=>m.group===group).map((m,n)=><fieldset className="metric" disabled={!session||busy} key={m.id}><legend><span className="metric-number">{String((group==='beauty'?5:0)+n+1).padStart(2,'0')}</span>{m.name}{draft.scores[m.id]&&<span className="metric-value">{draft.scores[m.id]} / 10</span>}</legend><p>{m.description}</p><div className="score-options">{Array.from({length:10},(_,i)=>i+1).map(score=><label className={draft.scores[m.id]===score?'chosen':''} key={score}><input type="radio" name={m.id} value={score} checked={draft.scores[m.id]===score} onChange={()=>edit({...draftRef.current,scores:{...draftRef.current.scores,[m.id]:score}})} aria-label={`${m.name}: ${score} out of 10`}/><span>{score}</span></label>)}</div><div className="score-anchors"><span>{m.low}</span><span>{m.high}</span></div></fieldset>)}</div>
            {group==='correctness'?<button className="category-next" onClick={()=>setGroup('beauty')}>Continue to Beauty <span>{METRICS.filter(m=>m.group==='beauty'&&draft.scores[m.id]).length}/5 rated</span></button>:<div className="comment-field"><label htmlFor="comment">Anything else? <span>Optional</span></label><textarea id="comment" value={draft.comment} disabled={!session||busy} maxLength={2000} placeholder="Share a detail that influenced your ratings…" onChange={e=>edit({...draftRef.current,comment:e.target.value})}/></div>}
          </div>
          <div className="rating-footer"><div className="save-status" role="status">{status.includes('saved')?<CheckCircle2 size={15}/>:<Save size={15}/>}<span>{session?(status||'Select a score to begin'):'Begin the study to save ratings'}</span></div><div className="footer-actions"><button className="secondary back" disabled={!session||index===0||busy} onClick={()=>{void navigate(index-1);}}><ChevronLeft size={16}/>Previous</button><button className="primary" disabled={!session||!currentComplete||busy} onClick={next}>{busy?'Saving…':index===total-1?'Review & finish':'Save & next diagram'}</button></div>{session&&!currentComplete&&<small className="remaining-note">Rate all 10 dimensions to continue. {10-count} remaining.</small>}</div>
        </section>
      </div>
      <footer className="study-bottom"><span>DiagramLab <span>·</span> Human evaluation study</span><span>1 = very poor · 10 = excellent</span></footer>
    </main>}
    {expanded&&<div className="modal-backdrop" role="presentation" onClick={()=>setExpanded(false)}><section className="image-modal" role="dialog" aria-modal="true" aria-label="Expanded diagram" onClick={e=>e.stopPropagation()}><button autoFocus className="modal-close" aria-label="Close expanded diagram" onClick={()=>setExpanded(false)}><X/></button><img src={diagram.image} alt="Expanded diagram"/></section></div>}
    {review&&<div className="modal-backdrop"><section className="review-modal" role="dialog" aria-modal="true" aria-labelledby="review-title"><div className="finish-icon"><CheckCircle2 size={28}/></div><p className="eyebrow">FINAL CHECK</p><h2 id="review-title">Ready to submit your study?</h2><p>{completed} of {total} diagrams are fully rated. You can review any diagram before submitting. Submission locks your responses.</p>{completed<total&&<p className="inline-error">Please finish every diagram before submitting.</p>}<div className="review-actions"><button className="secondary" onClick={()=>setReview(false)}>Keep reviewing</button><button className="primary" disabled={busy||completed!==total} onClick={submit}>{busy?'Submitting…':'Submit study'}</button></div></section></div>}
  </>;
}
