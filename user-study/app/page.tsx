import StudyClient from '../components/study/study-client';
import { STUDY, DIAGRAMS, publicDiagram } from '../lib/catalog';
export default function Home() {
  return <StudyClient preview={publicDiagram(DIAGRAMS[0])} sampleSize={STUDY.sampleSize} demo={STUDY.mode === 'demo'} />;
}
