# VWAP Gravitation Analysis — Project Structure

## Goal
Test whether price statistically gravitates back toward the *previous day's*
New York session end-of-day VWAP, more than random chance would predict.

## Pipeline (run in this order)

| Script | Input | Output | Purpose |
|---|---|---|---|
| `01_inspect_and_trim.py` | `data-2020.csv` (raw) | `nq_subset.csv` | Load raw data, verify timestamps, trim to a workable date range |
| `02_tag_sessions.py` | `nq_subset.csv` | `nq_tagged.csv` | Label every candle with its session (Asia/London/NY) and trading day |
| `03_compute_vwap.py` | `nq_tagged.csv` | `nq_vwap.csv` | Compute running VWAP + SD band per session, resetting at each session boundary |
| `04_extract_anchors.py` | `nq_vwap.csv` | `daily_anchors.csv` | Pull each day's frozen NY end-of-day VWAP (V) and SD (σ) |
| `05_containment_test.py` | `nq_vwap.csv` + `daily_anchors.csv` | `containment_results.csv` | For each day n+1, test whether price stayed within [V_n − σ_n, V_n + σ_n] |
| `06_report.py` | `containment_results.csv` | `final_report.md` + chart | Aggregate stats, compare to random-walk baseline, run significance test |

## Key definitions
- **V_n**: the NY session's VWAP value at its very last candle on day n (frozen anchor)
- **σ_n** (SD_n): the NY session's volume-weighted standard deviation at that same last candle
- **Containment band**: `[V_n − σ_n, V_n + σ_n]` — the price range we test day n+1 against
- **Null hypothesis**: price on day n+1 has no special relationship to V_n; any containment we see is what a random walk would produce anyway

## Status
- [x] Stage 1 built
- [ ] Stage 2
- [ ] Stage 3
- [ ] Stage 4
- [ ] Stage 5
- [ ] Stage 6 (report)
