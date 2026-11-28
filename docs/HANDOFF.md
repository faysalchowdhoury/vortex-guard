# VortexGuard: work history and resume guide

Last updated: 2026-11-28. Read this file first when resuming. It records the goal,
everything built so far, what was verified, problems hit, and the exact next steps.

## 1. Goal (original request)

Produce a complete, submission-ready 11-page two-column IEEE manuscript
(`\documentclass[conference]{IEEEtran}`) in `paper/main.tex` + `paper/references.bib`,
with every table filled from `results/metrics.json` by `scripts/populate_tex_tables.py`
and 7 figures in `figures/output/`:

1. `fig1_model_architecture.png` (dual-branch detector)
2. `fig2_end_to_end_pipeline.png` (probe, tokenizer, scorer, seccomp, freeze, rollback)
3. `fig3_scaling_tail_latency.png` (P50/P99 update latency and peak memory vs history length L)
4. `fig4_containment_timeline.png` (multi-stage exploit, score trace, tightening, freeze, rollback)
5. `fig5_pareto_tradeoff_radar.png` (runtime overhead % vs Macro-F1 Pareto + multi-metric radar)
6. `fig6_attack_category_heatmap.png` (per-family recall and FPR@99TPR, 6 ADFA-LD families)
7. `fig7_ablation_and_rollback_cdf.png` (ablation + 1,000-trial containment latency CDF)

Rules from the user:
- Sole author: "Faysal Chowdhoury", Rochester Institute of Technology / Independent Systems &
  Security Research, Bethesda, MD. (This overrides the usual two-author default block.)
- No em-dashes anywhere in `paper/main.tex`, `paper/references.bib`, `README.md`.
- No placeholders, no TODOs, no AI co-author trailers. Commits authored as
  `Faysal Chowdhoury <113149559+faysalchowdhoury@users.noreply.github.com>`, no trailers.
- No PINNs. 25 to 30 real, verifiable BibTeX references (eBPF, OverlayFS, seccomp, Mamba/S4,
  ADFA-LD, HIDS, LLM/agent sandboxing).
- Section plan: I Intro (~1.5 p, 4 contributions, headline numbers), II Background and threat
  model (~1 p), III Architecture and math (~2.5 p: Mamba recurrence h_t = Abar h_{t-1} + Bbar x_t,
  graph divergence operator, gated fusion), IV Kernel and sandbox implementation (~1.5 p),
  V Evaluation (~3 p), VI Security analysis, limitations, related work (~1 p: gVisor,
  Firecracker, Falco, Transformer HIDS), VII Conclusion (~0.5 p, repo link).
- Compile with pdflatex/bibtex (MiKTeX is installed on Windows), check 10 to 12 pages,
  no `??`, no overfull figure boxes. Push to `main` of
  https://github.com/faysalchowdhoury/vortex-guard (currently a PRIVATE repo created by Claude;
  the user may make it public).

Integrity rule adopted: every number in the paper must come from a real run. Nothing is
estimated or invented. If a result is worse than hoped (for example rollback not under 5 ms),
the paper reports the measured value.

## 2. What exists and is verified

| Component | File | Status |
|---|---|---|
| Syscall tables (i386 for ADFA-LD, x86_64 live) | `vortexguard/syscall_tables.json` | generated from kernel headers |
| Multi-channel tokenizer (token, category, flux edge; x86_64 to i386 by name) | `vortexguard/tokenizer.py` | unit tested |
| ADFA-LD leakage-free split | `vortexguard/data.py` | done, stats in `results/split.json` |
| VortexGuard model (selective SSM branch + flux-divergence branch + gate), baselines | `vortexguard/models.py` | unit tested: streaming O(1) step equals batch forward |
| Online scorers (VG/LSTM recurrent, Transformer window re-encode) | `vortexguard/scorers.py` | written, not yet run |
| CO-RE eBPF ring-buffer probe + loader | `kernel/ebpf_syscall_probe.bpf.c`, `kernel/ebpf_syscall_probe.c` | built and verified on WSL2 kernel 6.6 (fork tracking, 0 drops) |
| seccomp static allowlist + USER_NOTIF dynamic tightening + uid drop | `kernel/vg_sandbox_exec.c` | built and verified |
| OverlayFS sandbox (SIGSTOP freeze, lazy unmount, renameat2 RENAME_EXCHANGE swap, remount) | `vortexguard/overlay_sandbox.py` | written, not yet run |
| Containment daemon | `vortexguard/daemon.py` | written, not yet run |
| Agent workloads (benign + 5-stage exploit, low-and-slow, mimicry) | `scripts/workloads/agent_task.py` | written, not yet run |
| Training/eval matrix | `scripts/train_eval.py`, `scripts/run_detection.sh` | only static baseline finished |
| Overhead benchmark | `scripts/bench_overhead.py`, `scripts/workloads/bench_workload.py` | written, not yet run |
| Scaling benchmark | `scripts/bench_scaling.py` | written, not yet run |
| Aggregation to metrics.json | `scripts/aggregate_metrics.py` | written, not yet run |
| Tests | `tests/test_core.py` | 3 passing |

Not yet written: `scripts/make_figures.py` (all 7 figures), `scripts/populate_tex_tables.py`,
`paper/main.tex`, `paper/references.bib`, final README.

### Results so far

- Split (`results/split.json`): train 833 normal / 356 attack (30 attack instances);
  val 1,183 / 148 (12 instances); test 1,716 / 235 (18 instances). Attack instances split by
  folder per family (1-5 train, 6-7 val, 8-10 test). Exact-duplicate removal dropped
  129 val traces and **1,351 test traces** that were byte-identical to train/val traces
  (worth reporting: naive random splits of ADFA-LD leak heavily).
- `results/runs/StaticSeccompRuleBaseline_seed0.json`: static allowlist baseline scored.
- No neural model trained yet.

## 3. Key design decisions

- ADFA-LD has no process tree or socket metadata, so on ADFA-LD the flux-divergence branch
  uses syscall-derived edge fluxes on a 5-node resource graph (P process, F files, N network,
  C children, K privilege; edges F->P, P->F, N->P, P->N, P->C, P->K). Divergence = B phi with
  B the node-edge incidence matrix; fluxes are leaky integrals at tau = 16 and 512 events.
  Say this plainly in the paper. In live mode the same features come from eBPF events.
- Model: d_model 64, 2 selective SSM layers, expand 2, d_state 8, conv 4, dt_rank 4;
  flux branch hidden 32; gate g = sigmoid(w^T[z_ssm; z_flux] + b); logit = g s_ssm + (1-g) s_flux;
  training loss BCE(fused) + 0.5 (BCE(ssm) + BCE(flux)), AdamW lr 2e-3, wd 1e-2.
- Training: traces truncated to L_max = 512 (above the median length), token-budget batching
  (<= 32 traces and <= 4,096 padded tokens per batch), max 12 epochs, early stop on val AUPRC
  (patience 3), 3 seeds. Thresholds picked on validation (max F1). Static baseline uses its
  fixed rule (any syscall outside the profile = violation).
- Static seccomp baseline profile = syscalls seen in benign training traces (how container
  profiles are usually generated by tracing).
- Transformer scorer in live mode re-encodes the last L events per update (no recurrent state);
  scaling benchmark runs each cell under a 1.5 GB RLIMIT_DATA "per-tenant budget" so
  over-budget cells fail cleanly instead of crashing the host.
- Live containment thresholds are calibrated on benign live runs only (no attack data):
  tau_soft = max benign score (triggers seccomp tightening), tau_hard = tau_soft + 1 logit
  (triggers freeze + rollback). Revisit after the pilot if scores do not separate; if the
  ADFA-trained model does not transfer to modern x86_64 traces, report that and consider a
  live-trace fine-tune with seed-disjoint splits.

## 4. Problems hit and fixes (do not repeat)

1. **WSL2 PID namespace**: `fork()` PIDs in WSL differ from kernel global TGIDs. The probe now
   resolves the root by its namespace-local TGID via CO-RE
   (`task->group_leader->thread_pid->numbers[level].nr`) and then tracks global TGIDs.
   `bpf_get_ns_current_pid_tgid` did not match on WSL.
2. **Quadratic backward** in the Python selective scan: per-step slicing `dA[:, t]` made autograd
   allocate full-size gradients every step. Fixed with `unbind(1)`.
3. **Host out of memory**: the PC has 8 GB; WSL took 2.7 GB+, Windows raised "Out of Virtual
   Memory", and WSL crashed three times (`Wsl/Service/E_UNEXPECTED`). Batch 32 x 1024 tokens
   needed >3 GB. Now: one sequential job, ~0.8 GB peak. Before resuming, make sure Windows shows
   >= 2.5 GB "Available", and (with the user's OK) add `C:\Users\sheel\.wslconfig`:
   ```
   [wsl2]
   memory=2560MB
   swap=4GB
   ```
   then `wsl --shutdown`.
4. **Quoting**: `wsl -- bash -lc '...'` expands `$(...)`/`$VAR` early. Put scripts in files.
   From Git Bash, prefix `MSYS_NO_PATHCONV=1` when passing `/mnt/...` paths to `wsl`.
5. Ubuntu 26.04 uses uutils coreutils: use `tail -n 2`, not `tail -2`.
6. `pkill -f train_eval` inside a `bash -c` that contains that string kills itself.
7. Keep a `wsl.exe` session attached (run the job from a foreground or tool-managed background
   command); detached jobs died when the VM went idle or crashed.

## 5. Environment to rebuild (all local copies were deleted on 2026-11-28 to free the PC)

WSL Ubuntu 26.04, kernel 6.6.87.1-microsoft-standard-WSL2, BTF at /sys/kernel/btf/vmlinux.
Run as root inside WSL (passwordless sudo works).

```bash
sudo apt-get install -y clang llvm libbpf-dev libelf-dev zlib1g-dev make gcc pkg-config \
     python3-venv libseccomp-dev git bpftool
python3 -m venv ~/vg-venv
~/vg-venv/bin/pip install --index-url https://download.pytorch.org/whl/cpu torch
~/vg-venv/bin/pip install numpy scikit-learn matplotlib pytest
id agent || useradd -M -s /bin/bash agent        # uid 1000 expected by the scripts

git clone https://github.com/faysalchowdhoury/vortex-guard  /mnt/c/Users/sheel/Downloads/vortex-guard
cd /mnt/c/Users/sheel/Downloads/vortex-guard
mkdir -p data && cd data
git clone --depth 1 https://github.com/verazuo/a-labelled-version-of-the-ADFA-LD-dataset adfa_src
mkdir adfa && cd adfa && unzip -q ../adfa_src/ADFA-LD.zip && cd ../..
make -C kernel
~/vg-venv/bin/python -m pytest -q tests
```

Windows: MiKTeX pdflatex at `C:\Users\sheel\AppData\Local\Programs\MiKTeX\miktex\bin\x64\pdflatex`.
scholar-viz (user's figure skill): `pip install git+https://github.com/faysalchowdhoury/scholar-viz`
(use its IEEE style and critic checks for figures; install failed once only because WSL was down).

## 6. Remaining steps, in order

1. Detection matrix (about 3 to 4.5 h): `PY=~/vg-venv/bin/python bash scripts/run_detection.sh`
   (resumes; finished runs in `results/runs/` are skipped). Optional time saver: 2 seeds for the
   three ablations.
2. Live pilot: `sudo ~/vg-venv/bin/python scripts/containment_trials.py --phase pilot`
   (check score separation, scoring throughput in us/event).
3. Calibrate: `--phase calibrate --n 50`; trials: `--phase trials --n 1000 --n-benign 200`
   (about 1 to 1.5 h). Overlay must live on the WSL ext4 fs (`/var/lib/vortexguard`), not /mnt/c.
4. `sudo ~/vg-venv/bin/python scripts/bench_overhead.py` (about 30 to 45 min, machine idle).
5. `~/vg-venv/bin/python scripts/bench_scaling.py` (about 20 min).
6. `~/vg-venv/bin/python scripts/aggregate_metrics.py` -> `results/metrics.json`.
7. Write `scripts/make_figures.py` (7 figures from metrics.json and run JSONs; fig1/fig2 are
   diagrams), `scripts/populate_tex_tables.py` (fills tables between marker comments in
   main.tex and defines `\newcommand` macros for every number used in prose), `paper/main.tex`,
   `paper/references.bib`, README. Use the `/papercraft` and `scholar-humanize` skills for prose.
8. Compile, check page count and `??`, grep for em-dashes, commit as Faysal Chowdhoury, push.

Total remaining wall clock: roughly 6 to 8 hours, mostly unattended.
