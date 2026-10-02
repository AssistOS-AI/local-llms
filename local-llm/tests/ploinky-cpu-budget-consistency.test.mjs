import test from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs';
import path from 'node:path';
import os from 'node:os';
import { readSnapshot } from '../src/controller/hardware.mjs';
import { admit } from '../src/controller/admission.mjs';
import { observeMemoryBudget } from '../src/controller/ploinkyBudget.mjs';
import { createController } from '../src/controller/deployments.mjs';
import { createStateStore } from '../src/controller/stateStore.mjs';
import { loadSeedCatalog } from '../src/controller/catalog.mjs';
import { RUNNERS } from '../src/runners/index.mjs';
import { GPU_TEST_LOCK } from './ploinkyGpuFixture.mjs';
const GIB = 1024 ** 3;
const CAP = 16 * GIB;
const SEED = loadSeedCatalog();
const SMALL = SEED.find((e) => e.id === 'qwen2.5-0.5b-instruct-q4_k_m');
const runner = RUNNERS['llama.cpp'];
const params = runner.normalizeParams({}, {model: SMALL, profile: 'cpu'});
function reader(firstCurrent = 'malformed') {
  let currentCalls = 0;
  return {
    readFileSync(file) {
      if (file === '/proc/meminfo') return 'MemTotal: 134217728 kB\nMemAvailable: 125829120 kB\n';
      if (file === '/sys/fs/cgroup/memory.max') return String(CAP);
      if (file === '/sys/fs/cgroup/memory.current') return ++currentCalls % 2 ? firstCurrent : String(CAP);
      if (file === '/sys/fs/cgroup/cpu.max') return 'max 100000';
      if (file === '/proc/self/status') return 'Cpus_allowed_list:\t0-3\n';
      throw Object.assign(new Error('fixture absent'), {code:'ENOENT'});
    }
  };
}
function snapshot(fsApi) {
  return readSnapshot({dataDir:'/fixture',env:{},fsApi,statfs:async()=>({bavail:100000000,bsize:4096,blocks:200000000}),
    execFileImpl(command,args,options,cb) { cb(Object.assign(new Error('absent'),{code:'ENOENT'}),'',''); }});
}

test('public CPU admission refuses unreadable or zero headroom from one captured cgroup read', async (t) => {
  for (const firstCurrent of ['malformed', String(CAP)]) {
    const snap = await snapshot(reader(firstCurrent));
    const result = admit({runner,model:SMALL,source:SMALL.sources.gguf,params,snapshot:snap,profile:'cpu',decision:{cause:'absent',reason:'fixture'}});
    t.diagnostic(JSON.stringify({firstCurrent,rawCgroup:snap.cgroupMemory,observedBudget:snap.memoryBudget,admission:result.status}));
    assert.equal(result.status,'insufficient-now');
  }
});

for (const mode of ['unreadable-current', 'zero-headroom', 'overflow-max']) test(`actual controller never launches CPU runner with ${mode}`, async (t) => {
  const root = fs.mkdtempSync(path.join(os.tmpdir(),'llv-cpu-'));
  const dataDir = path.join(root,'data');
  const weights = path.join(root,'weights.gguf');
  fs.writeFileSync(weights,'x');
  const started=[];
  const fsApi=reader(mode === 'zero-headroom' ? String(CAP) : 'malformed');
  if (mode === 'overflow-max') {
    await snapshot(fsApi);
    const read = fsApi.readFileSync.bind(fsApi);
    fsApi.readFileSync = (file) => file === '/sys/fs/cgroup/memory.max' ? '9'.repeat(400) : read(file);
  }
  const controller=createController({dataDir,env:{PATH:'/usr/bin'},seedCatalog:SEED,runners:RUNNERS,stateStore:createStateStore({dataDir}),
    snapshot:()=>snapshot(fsApi),runnerLocks:GPU_TEST_LOCK,
    installer:{installable:()=>true,describe:async()=>({installed:true,runnable:true}),ensureRunnable:async()=>({rebuilt:false}),entryFor:(id)=>GPU_TEST_LOCK.runners[id],pathsFor:()=>({runDir:root})},
    download:async()=>({status:'complete',path:weights,bytesTransferred:0}),inspect:async()=>({state:'absent',bytes:0}),remove:async()=>0,
    fetchImpl:async()=>({ok:true,status:200,json:async()=>({})}),apiKeyFactory:()=> 'k'.repeat(43),
    detectRunner:(r)=>({installed:r.supported,version:'test',reason:null}),imageContract:null,fileExists:()=>true,sharedModelsRoot:null,
    readMemory:()=>({totalBytes:128*GIB,availableBytes:120*GIB}),readPressure:()=>0,
    readBudget:()=>observeMemoryBudget({maxText:String(CAP),currentText:String(CAP)}),unifiedGuardMs:60000,
    pollMs:2,stopGraceMs:20,
    startRunner(spec) {let resolve;let running=true;const h={...spec,pid:9999,exited:new Promise(r=>resolve=r),get running(){return running},async stop(){running=false;resolve({code:0,signal:'SIGTERM',error:null});return h.exited},async kill(){running=false;resolve({code:0,signal:'SIGKILL',error:null});return h.exited}};started.push(h);return h}
  });
  t.after(async()=>{await controller.stop();fs.rmSync(root,{recursive:true,force:true})});
  let accepted, rejection;
  try { accepted=await controller.run({runnerId:'llama.cpp',modelId:SMALL.id,requestId:'review-snapshot-inconsistent'}); } catch(error) {rejection=error.code;}
  const deadline=Date.now()+1000;
  while(!started.length && !rejection && Date.now()<deadline) await new Promise(r=>setTimeout(r,5));
  t.diagnostic(JSON.stringify({accepted,rejection,launches:started.length,phase:controller.state.deployment?.phase,admission:controller.state.deployment?.admission?.status}));
  assert.equal(started.length,0);
  assert.equal(rejection,'admission_insufficient_now');
});

test('overflowed memory.max retains established finite cap and refuses unknown instead of becoming unlimited', (t) => {
  const result=observeMemoryBudget({maxText:'9'.repeat(400),currentText:'malformed',established:CAP});
  t.diagnostic(JSON.stringify(result));
  assert.equal(result.memoryReadState,'unknown');
  assert.equal(result.finiteMemoryBytes,CAP);
});

test('public admission refuses an overflowing memory.max after a finite cap was established', async (t) => {
  let overflowing=false;
  const fsApi=reader();
  const read=fsApi.readFileSync.bind(fsApi);
  fsApi.readFileSync=(file)=>file==='/sys/fs/cgroup/memory.max'&&overflowing?'9'.repeat(400):read(file);
  const initial=await snapshot(fsApi);
  assert.equal(initial.memoryBudget.finiteMemoryBytes,CAP);
  overflowing=true;
  const snap=await snapshot(fsApi);
  const result=admit({runner,model:SMALL,source:SMALL.sources.gguf,params,snapshot:snap,profile:'cpu',decision:{cause:'absent',reason:'fixture'}});
  t.diagnostic(JSON.stringify({rawCgroup:snap.cgroupMemory,observedBudget:snap.memoryBudget,admission:result.status,reasonCode:result.reasonCode}));
  assert.equal(result.status,'insufficient-now');
  assert.equal(result.reasonCode,'budget_unreadable');
});

test('snapshot reads each cgroup memory file once and derives consistent raw and internal views', async () => {
  const calls = new Map();
  const fsApi = reader('0'); const read = fsApi.readFileSync.bind(fsApi);
  fsApi.readFileSync = (file) => { calls.set(file, (calls.get(file) || 0) + 1); return read(file); };
  const snap = await snapshot(fsApi);
  assert.equal(calls.get('/sys/fs/cgroup/memory.max'), 1);
  assert.equal(calls.get('/sys/fs/cgroup/memory.current'), 1);
  assert.equal(snap.cgroupMemory.currentBytes, 0);
  assert.equal(snap.memoryBudget.headroomBytes, CAP);
  assert.equal(Object.keys(snap).includes('memoryBudget'), false);
});
