export const METRICS = [
  { id: 'layout', name: 'Layout', group: 'correctness', description: 'Are elements placed in a sensible order, without unintended overlap or content outside the canvas?', low: 'Confusing or broken layout', high: 'Clear, well-organized layout' },
  { id: 'connectivity', name: 'Connectivity', group: 'correctness', description: 'Are the required connections present, with arrows joining the right elements in the right direction?', low: 'Missing or incorrect connections', high: 'All connections are correct' },
  { id: 'presence', name: 'Presence', group: 'correctness', description: 'Are all the components mentioned in the caption visibly present?', low: 'Most required elements are missing', high: 'Every required element is present' },
  { id: 'details', name: 'Details', group: 'correctness', description: 'Are labels complete, accurate, and meaningful, rather than missing or placeholders?', low: 'Missing or incorrect labels', high: 'Complete, accurate labels' },
  { id: 'legibility', name: 'Legibility', group: 'correctness', description: 'Is the text comfortably readable, without tiny type, clipping, or overflow?', low: 'Very difficult to read', high: 'Comfortably readable throughout' },
  { id: 'aesthetics', name: 'Aesthetics', group: 'beauty', description: 'How polished, visually coherent, and professionally styled is the diagram?', low: 'Unpolished and inconsistent', high: 'Polished and coherent' },
  { id: 'palette', name: 'Palette', group: 'beauty', description: 'How purposeful and coordinated is the use of color?', low: 'Poor or distracting color choices', high: 'Purposeful, coordinated colors' },
  { id: 'ink_balance', name: 'Ink balance', group: 'beauty', description: 'Does the canvas have a healthy amount of visual content, without feeling empty or overloaded?', low: 'Extremely empty or overloaded', high: 'A comfortable amount of visual content' },
  { id: 'density', name: 'Density', group: 'beauty', description: 'Are elements comfortably spaced, without cramped or crowded clusters?', low: 'Cramped or crowded', high: 'Comfortable, even spacing' },
  { id: 'balance', name: 'Balance', group: 'beauty', description: 'Is the composition balanced, with well-distributed visual weight and surrounding whitespace?', low: 'Strongly lopsided composition', high: 'Balanced composition and whitespace' },
] as const;
export type MetricId = typeof METRICS[number]['id'];
export type Scores = Partial<Record<MetricId, number>>;
export type PublicDiagram = { id: string; image: string; caption: string };
export type Rating = { scores: Scores; comment: string; durationMs: number; updatedAt?: string };
export type StudySession = { id: string; participantCode: string; studyVersion: string; mode: string; submittedAt: string | null; diagrams: PublicDiagram[]; ratings: Record<string, Rating> };
export const METRIC_IDS = METRICS.map(m => m.id);
export function isComplete(scores: Scores) { return METRIC_IDS.every(id => Number.isInteger(scores[id]) && scores[id]! >= 1 && scores[id]! <= 10); }
