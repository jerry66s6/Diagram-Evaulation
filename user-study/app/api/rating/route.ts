import {ApiError,assertEditable,db,handleError,json,payload,requireSession} from '../../../lib/server';
import {completeScores,validateRating} from '../../../lib/study-rules.mjs';
import type {Diagram} from '../../../lib/catalog';
export async function PUT(request:Request){try{const body=await payload(request),session=await requireSession(request);assertEditable(session);if(!(JSON.parse(session.assignment) as Diagram[]).some(d=>d.id===body.diagramId))throw new ApiError(403,'This diagram is not assigned to your session.');let rating;try{rating=validateRating(body);}catch(e){throw new ApiError(400,(e as Error).message);}const now=new Date().toISOString();
  const result=await db().prepare(`INSERT INTO study_ratings (session_id,diagram_id,scores,comment,duration_ms,complete,updated_at)
    SELECT ?,?,?,?,?,?,? WHERE EXISTS (SELECT 1 FROM study_sessions WHERE id=? AND submitted_at IS NULL)
    ON CONFLICT(session_id,diagram_id) DO UPDATE SET scores=excluded.scores,comment=excluded.comment,duration_ms=MAX(study_ratings.duration_ms,excluded.duration_ms),complete=excluded.complete,updated_at=excluded.updated_at
    WHERE EXISTS (SELECT 1 FROM study_sessions WHERE id=? AND submitted_at IS NULL)`).bind(session.id,body.diagramId,JSON.stringify(rating.scores),rating.comment,rating.durationMs,completeScores(rating.scores)?1:0,now,session.id,session.id).run();
  if(!result.meta.changes)throw new ApiError(409,'This study has already been submitted. Ratings are locked.');
  const saved=await db().prepare('SELECT duration_ms FROM study_ratings WHERE session_id=? AND diagram_id=?').bind(session.id,body.diagramId).first<{duration_ms:number}>();
  return json({rating:{...rating,durationMs:saved!.duration_ms,updatedAt:now}});
}catch(e){return handleError(e);}}
