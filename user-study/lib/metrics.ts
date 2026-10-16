export const METRICS = [
  { id: 'layout', name: 'Layout', group: 'correctness', description: 'Are elements placed in a sensible order, without unintended overlap, content outside the canvas, or arrows running through boxes or text?', low: 'Confusing or broken layout', high: 'Clear, well-organized layout' },
  { id: 'connectivity', name: 'Connectivity', group: 'correctness', description: 'Are the required connections present, with arrows joining the right elements in the right direction?', low: 'Missing or incorrect connections', high: 'All connections are correct' },
  { id: 'presence', name: 'Presence', group: 'correctness', description: 'Are all the components mentioned in the caption visibly present?', low: 'Most required elements are missing', high: 'Every required element is present' },
  { id: 'details', name: 'Details', group: 'correctness', description: 'Are labels complete, accurate, and meaningful, rather than missing or placeholders?', low: 'Missing or incorrect labels', high: 'Complete, accurate labels' },
  { id: 'legibility', name: 'Legibility', group: 'correctness', description: 'Is the text comfortably readable, without tiny type, clipping, overflow, or low contrast against its background?', low: 'Very difficult to read', high: 'Comfortably readable throughout' },
  { id: 'centering', name: 'Centering', group: 'beauty', description: 'Is the diagram centered on the canvas, each label centered in its shape, and neighboring elements evenly spaced?', low: 'Off-center or unevenly spaced', high: 'Centered, with even gaps' },
  { id: 'space_use', name: 'Space use', group: 'beauty', description: 'Are elements given room, rather than packed tightly while usable canvas space is left empty?', low: 'Crammed together while space is left empty', high: 'Space used comfortably' },
  { id: 'color_consistency', name: 'Color consistency', group: 'beauty', description: 'Do the colors form one coherent theme, with each kind of element keeping the same color?', low: 'Random or inconsistent colors', high: 'One coherent, consistent color theme' },
  { id: 'contrast', name: 'Contrast', group: 'beauty', description: 'Do boxes and arrows stand out clearly from the background? (Text contrast belongs to Legibility.)', low: 'Shapes and arrows fade into the background', high: 'Shapes and arrows stand out clearly' },
  { id: 'alignment', name: 'Alignment', group: 'beauty', description: 'Is the geometry deliberate: elements lined up or in a clear structure (such as a tree or ring), nothing slightly tilted, and arrows in one clean, consistent style?', low: 'Misaligned, tilted, or messy arrows', high: 'Neatly aligned, with clean, consistent arrows' },
  { id: 'style_uniformity', name: 'Style uniformity', group: 'beauty', description: 'Do elements with the same role share the same shape, size, border, font, and arrow style?', low: 'Mixed, inconsistent styles', high: 'Uniform style throughout' },
  { id: 'resolution', name: 'Resolution', group: 'beauty', description: 'Is the image crisp, without blur, pixelation, or compression artifacts?', low: 'Blurry or pixelated', high: 'Sharp and crisp' },
  { id: 'canvas_fill', name: 'Canvas fill', group: 'beauty', description: 'Do the diagram elements fill a reasonable part of the canvas, rather than being tiny in a mostly empty image?', low: 'Tiny elements in a mostly empty canvas', high: 'Elements fill the canvas well' },
] as const;
export type MetricId = typeof METRICS[number]['id'];
export type Scores = Partial<Record<MetricId, number>>;
export type PublicDiagram = { id: string; image: string; caption: string };
export type Rating = { scores: Scores; comment: string; durationMs: number; updatedAt?: string };
export type StudySession = { id: string; participantCode: string; participantName?: string; studyVersion: string; mode: string; submittedAt: string | null; diagrams: PublicDiagram[]; ratings: Record<string, Rating> };
export const METRIC_IDS = METRICS.map(m => m.id);
export function isComplete(scores: Scores) { return METRIC_IDS.every(id => Number.isInteger(scores[id]) && scores[id]! >= 1 && scores[id]! <= 10); }
