# Results and Methods Summary

This document summarizes the RL experiment in `results.ipynb` and explains key outputs.

## 1. Data cleaning and setup
- Source: `Clean Untitled spreadsheet - Data Tape.csv` from GitHub raw URL.
- Key extracted columns:
  - `InterestRatePct`: numeric percent parsed from `Loan Product`.
  - `Qualifying DTI`: numeric percent value.
  - `DPD`, `Loan Age`, `Original Loan Amount`, `Current Outstanding Principal` numeric conversion.
- Rows with NaN in core features dropped.
- Context state vector: `[InterestRatePct, DPD, Loan Age]`.

## 2. Phase 2: REINFORCE without baseline
- Policy: linear actor `mu = w2.dot(x)` with Gaussian action noise `sigma=0.2` clipped to [0,1].
- Reward: `allocation * InterestRatePct / 100`.
- Episode length: 4 steps.
- Update weight with full return G (no baseline).
- Output: learning curve (`returns2`) and mean return.

## 3. Phase 3: REINFORCE with baseline
- Added baseline `b3` (exponential moving average) and advantage `A = G - b3`.
- Expected effect: lower update variance, smoother return curve.
- Output: `returns3` and phase comparison plot.

## 4. Phase 4: DTI penalty + VaR/ES risk metrics
- Added `step4` with DTI violation penalty: if `Qualifying DTI > 0.20`, then reward minus `PENALTY_SCALE=0.01`.
- Risk metrics:
  - 5% VaR: `sorted(returns4)[ceil(0.05*n)-1]`.
  - 5% ES: mean of worst 5% returns.
- Output plots of `returns4`, `violations4`, and risk lines.

## 5. Post-training evaluation analysis
- On-policy Monte Carlo mean and 95% CI based on 1000 rollout samples.
- Mean return vs violations (error bars): demonstrates performance degradation with more compliance breaches.
- Return distribution histograms by violation count.
- 3D contour of policy value `mu` over `InterestRatePct` and `DPD` (loan age fixed median). Shows how learned allocation responds to risk/reward.

## 6. Key results
- Phase 2/3 comparison indicates baseline improves stability.
- Phase 4 risk-aware policy yields a tradeoff between overall return and DTI violations.
- VaR/ES provide downside risk control insights.
- Final on-policy estimate shows current policy expected return and CI.

## Usage
1. Install dependencies:
   `pip install -r requirements_2.txt`
2. Execute notebook:
   `jupyter nbconvert --to notebook --execute MainWorkingFile.ipynb --output results.ipynb`
3. Review plots and numbers in `results.ipynb` for evaluation.
