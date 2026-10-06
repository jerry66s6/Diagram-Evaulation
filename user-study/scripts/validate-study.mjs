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
balancedAssignment(config.diagrams,config.sampleSize);
console.log(`Catalog verified: ${config.diagrams.length} unique images, ${config.sampleSize} per session, ${config.mode} mode.`);
