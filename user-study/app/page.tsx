import StudyClient from '../components/study/study-client';
import { STUDY, SESSION_SIZE } from '../lib/catalog';
// Everyone sees the landing preview before starting, so it is a dataset diagram that is not in the study.
import preview from '../config/preview.json';
export default function Home() {
  return <StudyClient preview={{ id: preview.id, image: preview.image, caption: preview.caption }} sampleSize={SESSION_SIZE} demo={STUDY.mode === 'demo'} />;
}
