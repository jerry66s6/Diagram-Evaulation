import StudyClient from '../components/study/study-client';
import { STUDY, DIAGRAMS, SESSION_SIZE, publicDiagram } from '../lib/catalog';
// The landing preview is seen by everyone, so it is never one of the deliberately broken diagrams.
const PREVIEW = DIAGRAMS.find(d => d.label !== 'negative') ?? DIAGRAMS[0];
export default function Home() {
  return <StudyClient preview={publicDiagram(PREVIEW)} sampleSize={SESSION_SIZE} demo={STUDY.mode === 'demo'} />;
}
