import StudyClient from '../components/study/study-client';
import { STUDY, DIAGRAMS, SESSION_SIZE, publicDiagram } from '../lib/catalog';
export default function Home() {
  return <StudyClient preview={publicDiagram(DIAGRAMS[0])} sampleSize={SESSION_SIZE} demo={STUDY.mode === 'demo'} />;
}
