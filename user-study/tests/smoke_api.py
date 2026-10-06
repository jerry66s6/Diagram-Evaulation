"""Run against a local preview only. Creates demo sessions; never targets a hosted study."""
import csv, http.cookiejar, io, json, os, pathlib, sys, urllib.error, urllib.parse, urllib.request
base=sys.argv[1] if len(sys.argv)>1 else 'http://127.0.0.1:5173'
assert urllib.parse.urlparse(base).hostname in {'localhost','127.0.0.1','::1'}, 'Local preview only'
root=pathlib.Path(__file__).resolve().parents[1]
key=next(line.split('=',1)[1] for line in (root/'.env').read_text().splitlines() if line.startswith('STUDY_ADMIN_KEY='))
keys=['layout','connectivity','presence','details','legibility','aesthetics','palette','ink_balance','density','balance']
clients=[urllib.request.build_opener(urllib.request.HTTPCookieProcessor(http.cookiejar.CookieJar())) for _ in range(2)]
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
assert call('/api/session','POST',{'consent':False})[0]==400
status,result=call('/api/session','POST',{'consent':True});assert status==201,(status,result)
session=result['session'];assert len(session['diagrams'])==24
assert not any('difficulty' in d or 'source' in d for d in session['diagrams'])
assert call('/api/session','POST',{'consent':True})[1]['session']['id']==session['id']
assert call('/api/submit','POST',{})[0]==400
first=session['diagrams'][0]['id']
assert call('/api/rating','PUT',{'diagramId':first,'scores':{'layout':11},'comment':'','durationMs':0})[0]==400
assert call('/api/rating','PUT',{'diagramId':'not-assigned','scores':{},'comment':'','durationMs':0})[0]==403
assert call('/api/rating','PUT',{'diagramId':first,'scores':{},'comment':'','durationMs':0},extra={'Origin':'https://example.invalid'})[0]==403
for i,d in enumerate(session['diagrams']):
    status,result=call('/api/rating','PUT',{'diagramId':d['id'],'scores':{k:(i%10)+1 for k in keys},'comment':'=AUTOMATED TEST, "quoted"\nmultiline' if i==0 else 'Automated local verification','durationMs':1234+i})
    assert status==200,(status,result)
loaded=call('/api/session')[1]['session'];assert len(loaded['ratings'])==24
assert [d['id'] for d in loaded['diagrams']]==[d['id'] for d in session['diagrams']]
assert call('/api/submit','POST',{})[0]==200
assert call('/api/submit','POST',{})[0]==200
assert call('/api/rating','PUT',{'diagramId':first,'scores':{},'comment':'','durationMs':0})[0]==409
status,other=call('/api/session','POST',{'consent':True},client=1);assert status==201
assert other['session']['id']!=session['id'] and not other['session']['ratings']
status,data=call('/api/export?scope=completed',admin=True);assert status==200
rows=[r for r in csv.DictReader(io.StringIO(data)) if r['participant_code']==session['participantCode']]
assert len(rows)==24
assert {level:sum(r['difficulty']==level for r in rows) for level in ['easy','medium','hard']}=={'easy':8,'medium':8,'hard':8}
assert rows[0]['comment'].startswith("'=AUTOMATED TEST")
assert all(r['mode']=='demo' and r['rating_complete']=='1' for r in rows)
assert all(rows[i][k]==str((i%10)+1) for i in range(24) for k in keys)
assert call('/api/admin',admin=True)[1]['summary']['completed']>=1
status,data_all=call('/api/export?scope=all',admin=True)
partial=[r for r in csv.DictReader(io.StringIO(data_all)) if r['participant_code']==other['session']['participantCode']]
assert len(partial)==24 and all(r['layout']=='' for r in partial)
pathlib.Path('/private/tmp/diagram-study-verified-export.csv').write_text(data)
pathlib.Path('/private/tmp/diagram-study-test-session-ids.json').write_text(json.dumps([session['id'],other['session']['id']]))
print('PASS: consent, 24 × 10 ratings, exact 8/8/8 allocation, validation, autosave read-back, stable order, session isolation, submit lock, authenticated CSV, formula escaping and partial exports.')
