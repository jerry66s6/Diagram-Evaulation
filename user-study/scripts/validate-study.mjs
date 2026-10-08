import {readFileSync,existsSync} from 'node:fs';
import {resolve} from 'node:path';
import {createHash} from 'node:crypto';
import {balancedAssignment} from '../lib/study-rules.mjs';
const config=JSON.parse(readFileSync(new URL('../config/study.json',import.meta.url),'utf8'));
if(!['demo','research'].includes(config.mode))throw new Error('mode must be demo or research');
const hashes=new Set();
for(const d of config.diagrams){
  if(!/^\/diagrams\/[a-zA-Z0-9._-]+$/.test(d.image))throw new Error(`Invalid image path for ${d.id}`);
  if(!d.caption?.trim()||!d.source?.trim())throw new Error(`Missing caption or source for ${d.id}`);
  const path=resolve('public','.'+d.image);
  if(!existsSync(path))throw new Error(`Missing image for ${d.id}`);
  const digest=createHash('sha256').update(readFileSync(path)).digest('hex');
  if(digest!==d.sha256)throw new Error(`Image hash changed for ${d.id}; update the catalog intentionally.`);
  if(hashes.has(digest))throw new Error(`Duplicate image content at ${d.id}`);hashes.add(digest);
}
if(config.order==='sets'){
  for(let set=1;set<=config.sets;set++){
    const items=config.diagrams.filter(d=>d.set===set);
    if(items.length!==config.sampleSize)throw new Error(`Set ${set} has ${items.length} diagrams, not ${config.sampleSize}.`);
    const counts=['easy','medium','hard'].map(level=>items.filter(d=>d.difficulty===level).length);
    if(Math.max(...counts)-Math.min(...counts)>1)throw new Error(`Set ${set} is unbalanced: ${counts.join('/')}.`);
  }
  if(config.diagrams.some(d=>!(d.set>=1&&d.set<=config.sets)))throw new Error('Every diagram needs a set.');
}else if(config.order!=='fixed')balancedAssignment(config.diagrams,config.sampleSize);
console.log(`Catalog verified: ${config.diagrams.length} unique images, ${config.order==='sets'?`${config.sets} disjoint sets of ${config.sampleSize}`:config.order==='fixed'?`all ${config.diagrams.length} in fixed order`:`${config.sampleSize} per session`}, ${config.mode} mode.`);
