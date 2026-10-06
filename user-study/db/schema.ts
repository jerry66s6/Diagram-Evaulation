import { integer, text, sqliteTable, primaryKey, index } from 'drizzle-orm/sqlite-core';
export const sessions = sqliteTable('study_sessions', {
  id: text('id').primaryKey(),
  tokenHash: text('token_hash').notNull().unique(),
  participantCode: text('participant_code').notNull().unique(),
  studyVersion: text('study_version').notNull(),
  mode: text('mode').notNull(),
  assignment: text('assignment').notNull(),
  consentVersion: text('consent_version').notNull(),
  startedAt: text('started_at').notNull(),
  submittedAt: text('submitted_at'),
}, table=>[index('idx_sessions_version').on(table.studyVersion)]);
export const ratings = sqliteTable('study_ratings', {
  sessionId: text('session_id').notNull().references(()=>sessions.id),
  diagramId: text('diagram_id').notNull(),
  scores: text('scores').notNull(),
  comment: text('comment').notNull().default(''),
  durationMs: integer('duration_ms').notNull().default(0),
  complete: integer('complete').notNull().default(0),
  updatedAt: text('updated_at').notNull(),
},table=>[primaryKey({columns:[table.sessionId,table.diagramId]})]);
