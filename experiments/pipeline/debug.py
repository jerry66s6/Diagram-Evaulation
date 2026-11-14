"""Why the pipeline's accuracy is lower than with human scores, from saved results only (no API calls).

    python3 experiments/pipeline/debug.py

Needs results/, ../beauty_tuning/results/, and ../human_alignment/results/ (all git-ignored, kept locally).
"""
import os
os.chdir(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))
import csv, json, collections as C, numpy as np
from sklearn.metrics import roc_auc_score
from scipy.stats import spearmanr
H={d['id']:d for d in json.load(open('experiments/human_alignment/human_ratings.json'))['diagrams']}
COR=['layout','connectivity','presence','details','legibility']
BEA=['centering','space_use','color_consistency','contrast','alignment','style_uniformity','resolution','canvas_fill']
DIMS=COR+BEA; TUNED={'centering','space_use','alignment','canvas_fill'}
test=[f'HS{i}' for i in range(201,301)]
def judge(i,k):
    if k in TUNED: return json.load(open(f'experiments/beauty_tuning/results/{i}/rater_{k}.json'))['score']
    s=json.load(open(f'experiments/pipeline/results/{i}/{k}.json'))['score']; return s/10 if k in BEA else s
y=np.array([H[i]['label']=='negative' for i in test],int)
Hs=np.array([[H[i]['scores'][k] for k in DIMS] for i in test],float); Js=np.array([[judge(i,k) for k in DIMS] for i in test],float)
A=list(csv.DictReader(open('experiments/pipeline/predictions_beauty_tuned.csv')))      # human-trained GB, given judge / human scores
B={r['id']:r for r in csv.DictReader(open('experiments/pipeline/predictions_trained_on_judges_beauty_tuned.csv'))}
hum_call=np.array([int(r['called_bad_from_human_scores']) for r in A]); pipe_call=np.array([int(r['called_bad']) for r in A]); judge_call=np.array([int(B[i]['called_bad']) for i in test])
print('== 1. Calls on the 100 test diagrams')
for name,c in [('human scores, human-trained GB',hum_call),('judge scores, judge-trained GB',judge_call),('judge scores, human-trained GB (pipeline)',pipe_call)]:
    print(f'  {name}: accuracy {(c==y).mean():.2f}, bad caught {c[y==1].sum()}/53, good flagged {c[y==0].sum()}/47')
print('  bad diagrams humans caught but pipeline missed:',int(((hum_call==1)&(pipe_call==0)&(y==1)).sum()),'| good diagrams pipeline flagged but humans-model kept:',int(((hum_call==0)&(pipe_call==1)&(y==0)).sum()))
print('\n== 2. Judge minus human, mean per dimension, by group (test set)')
miss=(y==1)&(pipe_call==0); fa=(y==0)&(pipe_call==1)
print('  | dim | all bad | bad the pipeline missed | good it flagged | all good |')
for n,k in enumerate(DIMS):
    d=Js[:,n]-Hs[:,n]
    print(f'  | {k} | {d[y==1].mean():+.2f} | {d[miss].mean():+.2f} | {d[fa].mean():+.2f} | {d[y==0].mean():+.2f} |')
print('\n== 3. Inserted errors: caught by the dimension it targets (score < 8), all 300, correctness judge d17 vs humans')
rows=[]
for i,d in H.items():
    if d['label']!='negative': continue
    for e in d['defects']:
        k=e['criterion'].split('.')[0]
        if k not in COR: continue
        j=json.load(open(f'experiments/human_alignment/results/{i}/direct_d17_{k}.json'))['score']
        rows.append((e['defect'],k,d['scores'][k]<8,j<8))
g=C.defaultdict(lambda:[0,0,0])
for de,k,hc,jc in rows: g[(de,k)][0]+=1; g[(de,k)][1]+=hc; g[(de,k)][2]+=jc
for (de,k),(n,hc,jc) in sorted(g.items(),key=lambda kv:(kv[1][2]-kv[1][1])/kv[1][0]):
    print(f'  {de} ({k}): n={n} humans {hc} judge {jc}')
print('  total: humans',sum(v[1] for v in g.values()),'judge',sum(v[2] for v in g.values()),'of',sum(v[0] for v in g.values()))
print('\n== 4. Run-to-run: two independent runs of the same correctness prompt on HS201-HS300 (study d17 vs pipeline)')
for k in COR:
    a=np.array([json.load(open(f'experiments/human_alignment/results/{i}/direct_d17_{k}.json'))['score'] for i in test]); b=np.array([json.load(open(f'experiments/pipeline/results/{i}/{k}.json'))['score'] for i in test]); h=np.array([H[i]['scores'][k] for i in test])
    print(f'  {k}: identical {np.mean(a==b):.0%}, mean |diff| {np.abs(a-b).mean():.2f}, Spearman run-run {spearmanr(a,b).statistic:.2f}; vs human: run1 {spearmanr(a,h).statistic:.2f}, run2 {spearmanr(b,h).statistic:.2f}, average {spearmanr((a+b)/2,h).statistic:.2f}')
a=np.array([[json.load(open(f'experiments/human_alignment/results/{i}/direct_d17_{k}.json'))['score'] for k in COR] for i in test]).mean(1); b=Js[:,:5].mean(1)
print(f'  correctness mean, AUC good vs bad: run1 {roc_auc_score(y,-a):.3f}, run2 {roc_auc_score(y,-b):.3f}, average of runs {roc_auc_score(y,-(a+b)/2):.3f}; humans {roc_auc_score(y,-Hs[:,:5].mean(1)):.3f}')
print('\n== 5. By difficulty (judge-trained GB / pipeline / human)')
for lv in ['easy','medium','hard']:
    at=np.array([H[i]['difficulty']==lv for i in test])
    print(f'  {lv} n={at.sum()}: accuracy {(judge_call[at]==y[at]).mean():.2f} / {(pipe_call[at]==y[at]).mean():.2f} / {(hum_call[at]==y[at]).mean():.2f}')
