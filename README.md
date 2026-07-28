# Ramp Rate Independence and Grid-Facing Disturbance Characteristics of Co-Located AI Data Center Workloads

![IEEE IECON](https://img.shields.io/badge/IEEE%20IECON-2026-blue)
![License](https://img.shields.io/badge/License-MIT-yellow)
![Python](https://img.shields.io/badge/python-3.9+-blue)

Companion code, data pipeline, and extended results for the IECON 2026 submission testing whether co-located AI workloads on a shared GPU cluster ramp independently of one another, and what that independence implies for grid-facing disturbance and facility-level power management.

## Publication Status

| Stage | Status |
|---|---|
| Submission | ⬜ Not yet submitted |
| Peer Review | ⬜ Pending |
| Decision | ⬜ Pending |
| Camera-Ready | ⬜ Pending |
| IEEE Xplore | ⬜ Pending |

*Update this table as the paper progresses through review. Conference: IEEE IECON 2026, the 52nd Annual Conference of the IEEE Industrial Electronics Society, Doha, Qatar, October 18-21, 2026.*

## What This Repository Contains

This is not a results dump. Every number, table, and figure in the paper is produced directly by `ramp_analysis.py` running on the real, publicly released production measurements described below. There are no synthetic data points anywhere in this pipeline.

| File | Purpose |
|---|---|
| `ramp_analysis.py` | Full analysis pipeline: loads the real per-workload power traces, computes ramp rate statistics, runs the cross-workload independence test (pairwise correlation and variance superposition), and runs the ROCOF sensitivity sweep over assumed grid inertia. Regenerates Table I, the correlation matrix, and the core figures. |
| `requirements.txt` | Exact Python dependencies needed to run the analysis. |
| `ramp_main.tex` | The paper source, IEEE conference two-column format. |
| `ramp_references.bib` | Full bibliography, 20 references, each independently verified against the original published source. |

## Dataset

This work uses the publicly released production measurements accompanying:

> P. Colangelo *et al.*, "AI data centres as grid-interactive assets," *Nature Energy*, vol. 11, no. 2, pp. 254-261, 2026.

The dataset (a 256-GPU cluster in Phoenix, Arizona, running six concurrent LLaMA-8B and MPT workloads at one-minute resolution) is publicly available at:
`https://github.com/ai-emerald/emerald-ai-demo-may-2025`

Download that dataset's `data/` folder and place it alongside `ramp_analysis.py`, or pass its path directly:

```bash
python ramp_analysis.py --data-dir /path/to/emerald-ai-demo-may-2025/data/
```

## Running the Analysis

```bash
git clone https://github.com/iamyogesh2001/RampIndependence.git
cd RampIndependence
pip install -r requirements.txt
python ramp_analysis.py --data-dir /path/to/emerald/data/
```

This will print:
- Table I: per-workload ramp rate statistics (mean, P95, P99, max)
- The cross-workload correlation matrix summary (max and mean absolute Pearson r)
- The superposition test results (variance ratio, full trace and split-half validation)
- The ROCOF sensitivity table across assumed inertia constants H = 3 to 10 seconds

and will write reproducible figures to `./figures/` and a summary CSV of all statistics.

## Key Result

Co-located AI workloads on the same cluster ramp independently of one another. Cross-workload ramp correlations are uniformly weak (max |r| = 0.085), and aggregate ramp variance matches the sum of individual workload variances to within 3 percent, a result that remains stable when the measurement window is split in half. This independence is the empirical condition under which per-workload, diagonal power management (rather than full covariance-aware optimization) is a reasonable simplification for facility-level grid disturbance mitigation.

## Citation

If you use this code or build on this analysis, please cite:

```bibtex
@inproceedings{rethinapandian2026ramp,
  author    = {Rethinapandian, Yogesh and Sundararajan, Arun Karthik and Kumar, Kaushik and Prakash, Smrithi},
  title     = {Ramp Rate Independence and Grid-Facing Disturbance Characteristics of Co-Located {AI} Data Center Workloads},
  booktitle = {2026 IEEE 52nd Annual Conference of the Industrial Electronics Society (IECON)},
  year      = {2026}
}
```

## Acknowledgment

Large language model tools were used solely to assist in drafting the manuscript text and were not used to generate the research idea, methodology, or analysis. The conception of the paper and all simulations were carried out entirely by the authors. Results are derived from real measured data, not simulated or synthetic traces.

## License

This repository is released under the [MIT License](LICENSE).
