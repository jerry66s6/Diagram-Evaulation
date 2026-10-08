import { env } from 'cloudflare:workers';
import { createHash, randomBytes, timingSafeEqual } from 'node:crypto';
import { DIAGRAMS, FIXED_ORDER, SETS, STUDY, STUDY_VERSION, validateCatalog, type Diagram } from './catalog';
import { balancedAssignment, completeScores, shuffle } from './study-rules.mjs';
export type SessionRow={id:string;token_hash:string;participant_code:string;study_version:string;mode:string;assignment:string;consent_version:string;started_at:string;submitted_at:string|null;participant_name:string;set_number:number|null};
export type RatingRow={session_id:string;diagram_id:string;scores:string;comment:string;duration_ms:number;complete:number;updated_at:string};
export class ApiError extends Error {constructor(public status:number,message:string){super(message);}}
export function db(){if(!env.DB)throw new ApiError(503,'The study is temporarily unavailable. Please try again shortly.');return env.DB;}
export function json(value:unknown,status=200,headers:Record<string,string>={}) {return Response.json(value,{status,headers:{'Cache-Control':'no-store',...headers}});}
export function handleError(error:unknown) {if(error instanceof ApiError)return json({error:error.message},error.status);console.error('Study request failed',error);return json({error:'The study could not save or load your data. Your current inputs are still on this page; please try again.'},503);}
export function sameOrigin(request:Request){const origin=request.headers.get('origin');if(origin&&origin!==new URL(request.url).origin)throw new ApiError(403,'Cross-origin requests are not allowed.');}
export async function payload(request:Request){sameOrigin(request);if(!request.headers.get('content-type')?.includes('application/json'))throw new ApiError(415,'JSON is required.');const body=await request.text();if(body.length>16384)throw new ApiError(413,'Request is too large.');try{return JSON.parse(body);}catch{throw new ApiError(400,'Invalid JSON.');}}
function digest(token:string){return createHash('sha256').update(token).digest('hex');}
export async function findSession(request:Request){const token=request.headers.get('cookie')?.split(';').map(c=>c.trim()).find(c=>c.startsWith('diagram_study='))?.slice('diagram_study='.length);if(!token||!/^[a-f0-9]{64}$/.test(token))return null;return db().prepare('SELECT * FROM study_sessions WHERE token_hash = ? AND study_version = ?').bind(digest(token),STUDY_VERSION).first<SessionRow>();}
export async function requireSession(request:Request){const s=await findSession(request);if(!s)throw new ApiError(401,'Your study session was not found. Return to the study page to begin.');return s;}
export async function getRatings(id:string){const rows=await db().prepare('SELECT * FROM study_ratings WHERE session_id = ?').bind(id).all<RatingRow>();return rows.results;}
export async function publicSession(s:SessionRow){const ratings=await getRatings(s.id);return {id:s.id,participantCode:s.participant_code,participantName:s.participant_name,studyVersion:s.study_version,mode:s.mode,submittedAt:s.submitted_at,diagrams:(JSON.parse(s.assignment) as (Diagram&{description?:string})[]).map(({id,image,caption,description})=>({id,image,caption:caption??description??''})),ratings:Object.fromEntries(ratings.map(r=>[r.diagram_id,{scores:JSON.parse(r.scores),comment:r.comment,durationMs:r.duration_ms,updatedAt:r.updated_at}]))};}
export function participantName(value:unknown){const name=typeof value==='string'?value.trim().replace(/\s+/g,' '):'';if(!name)throw new ApiError(400,'Please enter your name to begin.');if(name.length>80)throw new ApiError(400,'Please keep your name under 80 characters.');return name;}
function random(){return randomBytes(4).readUInt32BE()/4294967296;}
// With fixed sets, take the lowest set no session holds yet. The unique index on
// (study_version, set_number) makes a set go to one session even when two people start at once.
async function insertSession(s:SessionRow){
  const insert=(row:SessionRow)=>db().prepare('INSERT INTO study_sessions (id,token_hash,participant_code,study_version,mode,assignment,consent_version,started_at,participant_name,set_number) VALUES (?,?,?,?,?,?,?,?,?,?)').bind(row.id,row.token_hash,row.participant_code,row.study_version,row.mode,row.assignment,row.consent_version,row.started_at,row.participant_name,row.set_number).run();
  if(!SETS){s.assignment=JSON.stringify(FIXED_ORDER?DIAGRAMS:balancedAssignment(DIAGRAMS,STUDY.sampleSize,random));await insert(s);return;}
  for(let set=1;set<=SETS;set++){
    const taken=await db().prepare('SELECT 1 FROM study_sessions WHERE study_version = ? AND set_number = ?').bind(STUDY_VERSION,set).first();
    if(taken)continue;
    s.set_number=set;s.assignment=JSON.stringify(shuffle(DIAGRAMS.filter(d=>d.set===set),random));
    try{await insert(s);return;}catch(e){if(!String((e as Error).message).includes('UNIQUE'))throw e;}
  }
  throw new ApiError(409,`All ${SETS} sets of diagrams have already been assigned. Please contact the researcher.`);
}
export async function newSession(request:Request,name:string){validateCatalog();const token=randomBytes(32).toString('hex'),id=crypto.randomUUID(),now=new Date().toISOString();const s:SessionRow={id,token_hash:digest(token),participant_code:'P-'+randomBytes(5).toString('hex').toUpperCase(),study_version:STUDY_VERSION,mode:STUDY.mode,assignment:'[]',consent_version:'named-ratings-v1',started_at:now,submitted_at:null,participant_name:name,set_number:null};await insertSession(s);return json({session:await publicSession(s)},201,{'Set-Cookie':`diagram_study=${token}; HttpOnly; SameSite=Lax; Path=/; Max-Age=2592000${new URL(request.url).protocol==='https:'?'; Secure':''}`});}
// Frees the set of an unfinished session (deleting its partial ratings) so the next participant gets it.
export async function releaseSession(participantCode:string){const s=await db().prepare('SELECT id FROM study_sessions WHERE participant_code = ? AND study_version = ? AND submitted_at IS NULL').bind(participantCode,STUDY_VERSION).first<{id:string}>();if(!s)throw new ApiError(404,'Only unfinished sessions of this study can be released.');await db().batch([db().prepare('DELETE FROM study_ratings WHERE session_id = ?').bind(s.id),db().prepare('DELETE FROM study_sessions WHERE id = ? AND submitted_at IS NULL').bind(s.id)]);}
export function requireAdmin(request:Request){const configured=env.STUDY_ADMIN_KEY || process.env.STUDY_ADMIN_KEY;const supplied=request.headers.get('x-study-admin-key')||'';if(!configured||configured.length<24)throw new ApiError(503,'Researcher access has not been configured.');const a=Buffer.from(digest(configured)),b=Buffer.from(digest(supplied));if(!timingSafeEqual(a,b))throw new ApiError(401,'The researcher access key is incorrect.');}
export function assertEditable(s:SessionRow){if(s.submitted_at)throw new ApiError(409,'This study has already been submitted. Ratings are locked.');}
export function fullyRated(rows:RatingRow[],assignment:Diagram[]){return assignment.every(d=>rows.some(r=>r.diagram_id===d.id&&completeScores(JSON.parse(r.scores))));}
