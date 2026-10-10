"""GPU provenance, exclusivity monitor, synchronized timing, strict validity."""
import os, platform, statistics, subprocess, threading, time
from pathlib import Path

class Contaminated(RuntimeError):pass

def smi():
    text=subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid,name,memory.used,utilization.gpu,power.limit,clocks.sm,temperature.gpu','--format=csv,noheader,nounits'],text=True,timeout=15)
    rows=[]
    for line in text.strip().splitlines():
        i,u,n,m,v,p,c,t=[x.strip() for x in line.split(',')]
        rows.append(dict(index=int(i),uuid=u,name=n,memory_mib=int(m),util=int(v),power_limit_w=float(p),sm_clock=c,temperature=t))
    return rows

def host_pids():
    # Map our container PID to the host PID when exposed via NSpid.
    pids={os.getpid()}
    for line in Path('/proc/self/status').read_text().splitlines():
        if line.startswith('NSpid:'):pids.update(int(x) for x in line.split()[1:])
    return pids

def empty(row):return 'A800' in row['name'] and row['memory_mib']<=256 and row['util']==0

def diagnostic_idle(row):return 'A800' in row['name'] and row['util']==0

class Guard:
    def __init__(self,uuid,allow_unverified=False,shared_idle_diagnostic=False):
        self.uuid=uuid;self.samples=[];self.errors=[];self.warnings=[];self.stop=threading.Event()
        before=next(r for r in smi() if r['uuid']==uuid)
        self.shared_idle_diagnostic=shared_idle_diagnostic
        if not (diagnostic_idle(before) if shared_idle_diagnostic else empty(before)):raise Contaminated('A800 does not meet the requested start condition')
        self.before=before
        import pynvml as nv
        nv.nvmlInit();self.nv=nv;self.handle=nv.nvmlDeviceGetHandleByUUID(uuid);self.own=host_pids()
        initial=nv.nvmlDeviceGetComputeRunningProcesses(self.handle)
        self.baseline_pids={p.pid for p in initial} if shared_idle_diagnostic else set()
        self.baseline_processes=[dict(pid=p.pid,used_gpu_memory_bytes=getattr(p,'usedGpuMemory',None)) for p in initial]
        if initial and not shared_idle_diagnostic:raise Contaminated('GPU process present before worker CUDA initialization')
        self.pid_visibility=False;self.allow_unverified=allow_unverified;self.context_reference=None;self.ledger_violations=0;self.measurement_started=False
        self.thread=threading.Thread(target=self.loop,daemon=True);self.context_pid_inferred=None
    def bind_cuda_context(self):
        # Before/after attribution is recorded as inferred, never PID-certified.
        import torch
        torch.cuda.init();torch.empty(1,device='cuda');torch.cuda.synchronize()
        procs=[p for p in self.nv.nvmlDeviceGetComputeRunningProcesses(self.handle) if p.pid not in self.baseline_pids]
        for _ in range(20):
            if procs:break
            time.sleep(.1);procs=self.nv.nvmlDeviceGetComputeRunningProcesses(self.handle)
        if len(procs)>1:raise Contaminated('Multiple GPU processes appeared during isolated context initialization')
        if procs and procs[0].pid not in self.own:
            if not self.allow_unverified:raise Contaminated('GPU PID namespace cannot be verified')
            self.context_pid_inferred=procs[0].pid
        self.thread.start();self.poll()

    def poll(self):
        nv=self.nv;mem=nv.nvmlDeviceGetMemoryInfo(self.handle);util=nv.nvmlDeviceGetUtilizationRates(self.handle)
        procs=nv.nvmlDeviceGetComputeRunningProcesses(self.handle)
        own=[p.pid for p in procs if p.pid in self.own];foreign=[p.pid for p in procs if p.pid not in self.own and p.pid!=self.context_pid_inferred and p.pid not in getattr(self,'baseline_pids',set())]
        if own:self.pid_visibility=True
        import torch
        reserved=torch.cuda.memory_reserved() if torch.cuda.is_initialized() else 0
        baseline_pids=getattr(self,'baseline_pids',set())
        baseline_memory=sum(getattr(p,'usedGpuMemory',0) for p in procs if p.pid in baseline_pids and isinstance(getattr(p,'usedGpuMemory',0),int) and 0<=getattr(p,'usedGpuMemory',0)<=mem.used)
        own_nvml_values=[getattr(p,'usedGpuMemory',None) for p in procs if (p.pid in self.own or p.pid==self.context_pid_inferred) and isinstance(getattr(p,'usedGpuMemory',None),int) and 0<=getattr(p,'usedGpuMemory',None)<=mem.used]
        own_nvml_memory=sum(own_nvml_values)
        own_accounted=own_nvml_memory if own_nvml_values else reserved
        row_extra=mem.used-own_accounted-baseline_memory
        if self.measurement_started and self.context_reference is not None:
            if row_extra>self.context_reference+(512<<20):self.ledger_violations+=1
            else:self.ledger_violations=0
            if self.ledger_violations>=3:
                message='unaccounted GPU memory exceeded own process ledger'
                if getattr(self,'shared_idle_diagnostic',False):
                    # Shared-card trials cannot certify device memory ownership.
                    # Retain the anomaly explicitly; formal runs still fail here.
                    if not hasattr(self,'warnings'):self.warnings=[]
                    if message not in self.warnings:self.warnings.append(message)
                elif message not in self.errors:self.errors.append(message)
        row=dict(memory_ledger_accounting='nvml_own_process_memory' if own_nvml_values else 'torch_reserved_fallback',own_process_nvml_bytes=own_nvml_memory,own_non_torch_bytes=own_nvml_memory-reserved,time=time.time(),memory_bytes=mem.used,util=util.gpu,compute_pids=[p.pid for p in procs],foreign_pids=foreign,baseline_process_memory_bytes=baseline_memory,own_torch_reserved_bytes=reserved,unaccounted_bytes=row_extra)
        self.samples.append(row)
        if foreign:self.errors.append('foreign GPU process appeared')
        # In namespaces where NVML suppresses process lists, we cannot certify
        # exclusivity after allocation. Such timing remains diagnostic only.
    def loop(self):
        while not self.stop.wait(.5):
            try:self.poll()
            except Exception as e:self.errors.append('monitor_error:'+type(e).__name__)
    def begin_measurement(self):
        import torch
        torch.cuda.synchronize()
        self.poll()
        self.context_reference=self.samples[-1]['unaccounted_bytes']
        self.measurement_started=True
        self.poll()

    def assert_clean(self):
        if self.errors:raise Contaminated(self.errors[-1])
    def close(self):
        self.stop.set();self.thread.join(timeout=2) if self.thread.is_alive() else None
        try:self.poll()
        except Exception as e:self.errors.append('monitor_error:'+type(e).__name__)
        return dict(empty_before=self.before,uuid=self.uuid,node=platform.node(),pid_visibility=self.pid_visibility,valid_exclusive=not self.errors and self.pid_visibility and not getattr(self,'shared_idle_diagnostic',False),valid_empty_card_single_gpu=not self.errors and not getattr(self,'shared_idle_diagnostic',False),valid_shared_gpu_diagnostic=not self.errors and getattr(self,'shared_idle_diagnostic',False),shared_idle_diagnostic=getattr(self,'shared_idle_diagnostic',False),baseline_processes=getattr(self,'baseline_processes',[]),monitor_policy='shared_gpu_diagnostic_preexisting_processes_recorded' if getattr(self,'shared_idle_diagnostic',False) else ('pid_monitor' if self.pid_visibility else 'empty_card_plus_memory_ledger'),pid_visibility_limitation=not self.pid_visibility,context_pid_inferred=self.context_pid_inferred,context_reference_bytes=self.context_reference,ledger_accounting='NVML full own-process memory when visible; Torch reserved fallback when suppressed',errors=self.errors,warnings=getattr(self,'warnings',[]),samples=self.samples)

def timing(fn,guard,warmup=10,repeats=30):
    import torch
    with torch.inference_mode():
        print('first_forward_started',flush=True)
        compile_start=time.perf_counter();out=fn();torch.cuda.synchronize();first=(time.perf_counter()-compile_start)*1000;del out
        print('first_forward_ms',round(first,3),flush=True)
        for _ in range(warmup):out=fn();del out
        torch.cuda.synchronize();guard.begin_measurement();guard.assert_clean();torch.cuda.reset_peak_memory_stats()
        allocated=torch.cuda.memory_allocated();reserved=torch.cuda.memory_reserved()
        print('warmup_complete; measurement_started',flush=True)
        gpu=[];wall=[]
        for sample_index in range(repeats):
            guard.assert_clean();torch.cuda.synchronize();start=torch.cuda.Event(enable_timing=True);end=torch.cuda.Event(enable_timing=True)
            t=time.perf_counter();start.record();out=fn();end.record();torch.cuda.synchronize();wall.append((time.perf_counter()-t)*1000);gpu.append(start.elapsed_time(end));del out
            if (sample_index+1)%max(1,repeats//3)==0:print('timed_samples',sample_index+1,'wall_median_ms',round(statistics.median(wall),3),flush=True)
        guard.assert_clean()
    def summary(x):
        s=sorted(x);percentile=lambda p:s[min(len(s)-1,round((len(s)-1)*p))]
        return dict(median_ms=statistics.median(x),mean_ms=statistics.mean(x),p90_ms=percentile(.9),p95_ms=percentile(.95),std_ms=statistics.stdev(x) if len(x)>1 else 0,min_ms=min(x),max_ms=max(x),samples_ms=x)
    return dict(cuda=summary(gpu),wall=summary(wall),first_call_ms=first,warmup=warmup,repeats=repeats,baseline_allocated_bytes=allocated,baseline_reserved_bytes=reserved,peak_allocated_bytes=torch.cuda.max_memory_allocated(),peak_reserved_bytes=torch.cuda.max_memory_reserved(),peak_increment_bytes=torch.cuda.max_memory_allocated()-allocated)
