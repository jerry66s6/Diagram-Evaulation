import catalog from '../config/study.json';
import { createHash } from 'node:crypto';
export type Difficulty = 'easy' | 'medium' | 'hard';
export type Diagram = { id: string; image: string; caption: string; difficulty: Difficulty; source: string; sha256: string; set?: number };
export const STUDY = catalog;
export const DIAGRAMS = catalog.diagrams as Diagram[];
export const STUDY_VERSION = catalog.id + '-' + createHash('sha256').update(JSON.stringify(catalog)).digest('hex').slice(0, 12);
// "fixed": every participant sees every diagram in catalog order. "sets": the diagrams are
// split into `sets` fixed sets of sampleSize, and each participant gets the next unused set,
// so no diagram goes to two participants. Otherwise each participant gets a random,
// difficulty-balanced sample of sampleSize diagrams.
export const ORDER = ((catalog as { order?: string }).order ?? 'random') as 'fixed' | 'sets' | 'random';
export const FIXED_ORDER = ORDER === 'fixed';
export const SETS = ORDER === 'sets' ? (catalog as { sets?: number }).sets ?? 0 : 0;
export const SESSION_SIZE = FIXED_ORDER ? DIAGRAMS.length : STUDY.sampleSize;
export function validateCatalog() {
  if (ORDER === 'sets') {
    for (let set = 1; set <= SETS; set++) if (DIAGRAMS.filter(d => d.set === set).length !== STUDY.sampleSize) throw new Error(`Set ${set} does not have ${STUDY.sampleSize} diagrams.`);
    if (!SETS || DIAGRAMS.some(d => !Number.isInteger(d.set) || d.set! < 1 || d.set! > SETS)) throw new Error('Every diagram needs a set from 1 to ' + SETS + '.');
  }
  if (ORDER === 'random' && (!Number.isInteger(STUDY.sampleSize) || STUDY.sampleSize < 20 || STUDY.sampleSize > 30)) throw new Error('The study must assign 20–30 diagrams.');
  const ids = new Set<string>();
  for (const d of DIAGRAMS) {
    if (ids.has(d.id) || !d.caption?.trim() || !/^\/diagrams\/[a-zA-Z0-9._-]+$/.test(d.image) || !['easy', 'medium', 'hard'].includes(d.difficulty)) throw new Error('Invalid diagram catalog.');
    ids.add(d.id);
  }
  if (ORDER !== 'random') return;
  const min = Math.floor(STUDY.sampleSize / 3), max = Math.ceil(STUDY.sampleSize / 3);
  const counts = ['easy','medium','hard'].map(level => DIAGRAMS.filter(d => d.difficulty === level).length);
  if (counts.some(n => n < min) || counts.filter(n => n >= max).length < STUDY.sampleSize % 3) throw new Error('Not enough diagrams for balanced allocation.');
}
export function publicDiagram(d: Diagram) { return {id: d.id, image: d.image, caption: d.caption}; }
