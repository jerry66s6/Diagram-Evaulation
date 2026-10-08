"""Run against a local preview only. Creates demo sessions; never targets a hosted study."""
import csv, http.cookiejar, io, json, os, pathlib, sys, urllib.error, urllib.parse, urllib.request
base=sys.argv[1] if len(sys.argv)>1 else 'http://127.0.0.1:5173'
assert urllib.parse.urlparse(base).hostname in {'localhost','127.0.0.1','::1'}, 'Local preview only'
root=pathlib.Path(__file__).resolve().parents[1]
key=next(line.split('=',1)[1] for line in (root/'.env').read_text().splitlines() if line.startswith('STUDY_ADMIN_KEY='))
catalog=json.loads((root/'config'/'study.json').read_text())
fixed=catalog.get('order')=='fixed'
sets=catalog.get('sets',0) if catalog.get('order')=='sets' else 0
size=len(catalog['diagrams']) if fixed else catalog['sampleSize']
keys=['layout','connectivity','presence','details','legibility','aesthetics','palette','ink_balance','density','balance']
new_client=lambda:urllib.request.build_opener(urllib.request.HTTPCookieProcessor(http.cookiejar.CookieJar()))
clients=[new_client() for _ in range(2)]
def call(path,method='GET',body=None,admin=False,client=0,extra=None):
    headers={'Content-Type':'application/json'}
    if admin:headers['x-study-admin-key']=key
    if extra:headers.update(extra)
    request=urllib.request.Request(base+path,data=json.dumps(body).encode() if body is not None else None,method=method,headers=headers)
    try:r=clients[client].open(request)
    except urllib.error.HTTPError as e:r=e
    raw=r.read().decode('utf-8-sig')
    return r.status, json.loads(raw) if 'application/json' in r.headers.get('Content-Type','') else raw
assert call('/api/admin')[0]==401
assert call('/api/export')[0]==401
assert call('/api/session','POST',{'consent':False,'name':'Smoke test A'})[0]==400
assert call('/api/session','POST',{'consent':True,'name':'  '})[0]==400
status,result=call('/api/session','POST',{'consent':True,'name':'Smoke test A'});assert status==201,(status,result)
session=result['session'];assert len(session['diagrams'])==size
if fixed:assert [d['id'] for d in session['diagrams']]==[d['id'] for d in catalog['diagrams']]
assert not any('difficulty' in d or 'source' in d for d in session['diagrams'])
assert session['participantName']=='Smoke test A'
assert call('/api/session','POST',{'consent':True,'name':'Smoke test A'})[1]['session']['id']==session['id']
assert call('/api/submit','POST',{})[0]==400
first=session['diagrams'][0]['id']
assert call('/api/rating','PUT',{'diagramId':first,'scores':{'layout':11},'comment':'','durationMs':0})[0]==400
assert call('/api/rating','PUT',{'diagramId':'not-assigned','scores':{},'comment':'','durationMs':0})[0]==403
assert call('/api/rating','PUT',{'diagramId':first,'scores':{},'comment':'','durationMs':0},extra={'Origin':'https://example.invalid'})[0]==403
for i,d in enumerate(session['diagrams']):
    status,result=call('/api/rating','PUT',{'diagramId':d['id'],'scores':{k:(i%10)+1 for k in keys},'comment':'=AUTOMATED TEST, "quoted"\nmultiline' if i==0 else 'Automated local verification','durationMs':1234+i})
    assert status==200,(status,result)
loaded=call('/api/session')[1]['session'];assert len(loaded['ratings'])==size
assert [d['id'] for d in loaded['diagrams']]==[d['id'] for d in session['diagrams']]
assert call('/api/submit','POST',{})[0]==200
assert call('/api/submit','POST',{})[0]==200
assert call('/api/rating','PUT',{'diagramId':first,'scores':{},'comment':'','durationMs':0})[0]==409
status,other=call('/api/session','POST',{'consent':True,'name':'Smoke test B'},client=1);assert status==201
assert other['session']['id']!=session['id'] and not other['session']['ratings']
status,data=call('/api/export?scope=completed',admin=True);assert status==200
rows=[r for r in csv.DictReader(io.StringIO(data)) if r['participant_code']==session['participantCode']]
assert len(rows)==size
counts={level:sum(r['difficulty']==level for r in rows) for level in ['easy','medium','hard']}
assert max(counts.values())-min(counts.values())<=1
assert rows[0]['comment'].startswith("'=AUTOMATED TEST")
assert all(r['mode']==catalog['mode'] and r['rating_complete']=='1' and r['participant_name']=='Smoke test A' for r in rows)
assert all(rows[i][k]==str((i%10)+1) for i in range(size) for k in keys)
assert call('/api/admin',admin=True)[1]['summary']['completed']>=1
status,data_all=call('/api/export?scope=all',admin=True)
partial=[r for r in csv.DictReader(io.StringIO(data_all)) if r['participant_code']==other['session']['participantCode']]
assert len(partial)==size and all(r['layout']=='' for r in partial)
if sets:
    # Fill every free set: each session gets a different set and no diagram twice; then the study is full.
    seen={d['id'] for d in session['diagrams']}|{d['id'] for d in other['session']['diagrams']}
    assert len(seen)==2*size
    extra=[]
    while True:
        clients.append(new_client())
        status,result=call('/api/session','POST',{'consent':True,'name':f'Smoke test {len(clients)}'},client=len(clients)-1)
        if status==409:break
        assert status==201,(status,result)
        ids={d['id'] for d in result['session']['diagrams']};assert not ids&seen and len(ids)==size;seen|=ids;extra.append(result['session'])
    admin=call('/api/admin',admin=True)[1]
    taken=[s['set_number'] for s in admin['sessions'] if s['set_number']]
    assert len(taken)==len(set(taken))==sets
    # Release the unfinished sessions; the next participant gets the lowest freed set.
    for s in [other['session'],*extra]:
        assert call('/api/admin?participant='+s['participantCode'],'DELETE',admin=True)[0]==200
    assert call('/api/admin?participant='+session['participantCode'],'DELETE',admin=True)[0]==404
    clients.append(new_client())
    status,again=call('/api/session','POST',{'consent':True,'name':'Smoke test again'},client=len(clients)-1);assert status==201
    assert {d['id'] for d in again['session']['diagrams']}=={d['id'] for d in other['session']['diagrams']}
    assert call('/api/admin?participant='+again['session']['participantCode'],'DELETE',admin=True)[0]==200
pathlib.Path('/private/tmp/diagram-study-verified-export.csv').write_text(data)
pathlib.Path('/private/tmp/diagram-study-test-session-ids.json').write_text(json.dumps([session['id'],other['session']['id']]))
print(f'PASS: consent and name, {size} × 10 ratings, {f"{sets} disjoint sets, full-study refusal and release" if sets else "fixed order" if fixed else "balanced allocation"}, validation, autosave read-back, stable order, session isolation, submit lock, authenticated CSV, formula escaping and partial exports.')
