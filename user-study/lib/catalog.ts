import catalog from '../config/study.json';
import { createHash } from 'node:crypto';
export type Difficulty = 'easy' | 'medium' | 'hard';
export type Diagram = { id: string; image: string; caption: string; difficulty: Difficulty; source: string; sha256: string };
export const STUDY = catalog;
export const DIAGRAMS = catalog.diagrams as Diagram[];
export const STUDY_VERSION = catalog.id + '-' + createHash('sha256').update(JSON.stringify(catalog)).digest('hex').slice(0, 12);
export function validateCatalog() {
  if (!Number.isInteger(STUDY.sampleSize) || STUDY.sampleSize < 20 || STUDY.sampleSize > 30) throw new Error('The study must assign 20–30 diagrams.');
  const ids = new Set<string>();
  for (const d of DIAGRAMS) {
    if (ids.has(d.id) || !d.caption?.trim() || !/^\/diagrams\/[a-zA-Z0-9._-]+$/.test(d.image) || !['easy', 'medium', 'hard'].includes(d.difficulty)) throw new Error('Invalid diagram catalog.');
    ids.add(d.id);
  }
  const min = Math.floor(STUDY.sampleSize / 3), max = Math.ceil(STUDY.sampleSize / 3);
  const counts = ['easy','medium','hard'].map(level => DIAGRAMS.filter(d => d.difficulty === level).length);
  if (counts.some(n => n < min) || counts.filter(n => n >= max).length < STUDY.sampleSize % 3) throw new Error('Not enough diagrams for balanced allocation.');
}
export function publicDiagram(d: Diagram) { return {id: d.id, image: d.image, caption: d.caption}; }
