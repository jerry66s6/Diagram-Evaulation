export const SCORE_KEYS = ['layout','connectivity','presence','details','legibility','aesthetics','palette','ink_balance','density','balance'];
export function shuffle(values, random = Math.random) {
  const result = [...values];
  for (let i=result.length-1;i>0;i--) {const j=Math.floor(random()*(i+1));[result[i],result[j]]=[result[j],result[i]];}
  return result;
}
export function balancedAssignment(diagrams, count, random = Math.random) {
  if(!Number.isInteger(count)||count<20||count>30) throw new Error('Assign between 20 and 30 diagrams.');
  if(new Set(diagrams.map(d=>d.id)).size!==diagrams.length) throw new Error('Diagram IDs must be unique.');
  const base=Math.floor(count/3), remainder=count%3;
  const groups=Object.fromEntries(['easy','medium','hard'].map(level=>[level,shuffle(diagrams.filter(d=>d.difficulty===level),random)]));
  const eligible=shuffle(Object.keys(groups).filter(level=>groups[level].length>base),random);
  if(eligible.length<remainder||Object.values(groups).some(items=>items.length<base)) throw new Error('Insufficient diagrams for a balanced assignment.');
  const extra=new Set(eligible.slice(0,remainder));
  return shuffle(Object.entries(groups).flatMap(([level,items])=>items.slice(0,base+(extra.has(level)?1:0))),random);
}
export function validateRating(value) {
  if(!value || typeof value!=='object'||Array.isArray(value)||!value.scores||typeof value.scores!=='object'||Array.isArray(value.scores))throw new Error('A rating object is required.');
  for(const [key,score] of Object.entries(value.scores))if(!SCORE_KEYS.includes(key)||!Number.isInteger(score)||score<1||score>10)throw new Error('Scores must be integers from 1 to 10 for the ten study dimensions.');
  if(typeof value.comment!=='string'||value.comment.length>2000)throw new Error('Comments must be at most 2,000 characters.');
  if(!Number.isInteger(value.durationMs)||value.durationMs<0||value.durationMs>86400000)throw new Error('Invalid active duration.');
  return {scores:value.scores,comment:value.comment,durationMs:value.durationMs};
}
export function completeScores(scores) {return SCORE_KEYS.every(key=>Number.isInteger(scores?.[key])&&scores[key]>=1&&scores[key]<=10);}
export function csvCell(value) {
  let text=value==null?'':String(value);
  if(typeof value==='string' && /^\s*[=+\-@]/.test(text))text="'"+text;
  return '"'+text.replaceAll('"','""')+'"';
}
