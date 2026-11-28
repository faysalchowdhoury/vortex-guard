# VortexGuard

Zero-trust runtime sandboxing for autonomous AI agents executing untrusted code in multi-tenant clouds.

Status: work in progress. Read [docs/HANDOFF.md](docs/HANDOFF.md) for the full work history and resume steps.

Components:

- `kernel/`: libbpf CO-RE eBPF ring-buffer syscall probe
- `vortexguard/`: syscall tokenizer, selective state-space (Mamba) detector, process-tree and socket-flux divergence branch, seccomp tightening, OverlayFS snapshot rollback
- `scripts/`: ADFA-LD evaluation, benchmarks, figure generation, LaTeX table population
- `results/`, `figures/output/`, `paper/`: measured metrics, figures, and manuscript

## Data

Experiments use the ADFA-LD dataset (Creech and Hu, 2013). It is not redistributed here. Download it into `data/adfa/`.

## Author

Faysal Chowdhoury
