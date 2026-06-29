# Trust Score Audit Report

## 1. Objective
This report details the evaluation of the continuous ensemble trust scoring system built for the Redrob Hackathon. The score aggregates consistency, plausibility, identity, and assessment verification checks into a calibrated 0.0-1.0 risk probability.

> **CRITICAL ARCHITECTURAL NOTE**
> The trust score NEVER implements an auto-reject gate. It provides a continuous flag and exposes its breakdown for downstream components to "flag for review", never for silent exclusion.

## 2. Detection Rate on Known Honeypots
The master context documents an ~80-candidate honeypot population characterized by either an explicit `years_of_experience` vs. tenure mismatch or claiming multiple "expert" skills with zero duration.
- **Result:** The exact rule extraction identified **70** known honeypots in the dataset.
- The ensemble trust score successfully aggregates these signals to push these candidates towards a high risk score.

## 3. False Positive Analysis
A candidate is considered a "false positive" if their trust score exceeds 0.4 but they do not match the rigid honeypot patterns defined above.
- **Total False Positives (Score >= 0.4):** 9 candidates.
- **Characterization:** Sampling these candidates reveals a common profile:
  - They lack identity verification (`ident_risk` = 1.0).
  - They have assessment data where they claimed "advanced" proficiency in specific skills but scored poorly (~20.0), resulting in a high delta (e.g., `assm_risk` ~ 0.99).
  - They have consistent YOE and duration claims, and minimal template reliance.
- **Conclusion:** These aren't malicious bots, but they are legitimate "over-claimers". Flagging them for recruiter review is appropriate, confirming the system's value beyond rigid honeypot detection.

## 4. False Negative Stress Test (Adversarial Data)
To test the score's robustness against sophisticated fraud, we constructed a synthetic adversarial profile. This profile fabricated a consistent story: padded `career_history` durations matching a fabricated YOE, and expert skill claims backed by fabricated duration months.
- **Result:** The adversarial profile **PASSED undetected** with a Trust Score of 0.000.
- **Explicit Statement:** The current trust score catches sloppy/inconsistent fraud; it **does NOT reliably catch sophisticated, internally-consistent fraud**. The lack of assessment data correctly applies zero penalty, preventing us from catching consistent fabrications unless external verification is forced.

## 5. Bias Analysis
We analyzed whether the trust score disproportionately flags candidates with nonlinear career paths.
- **Correlation with Career Gaps:** 0.000
- **Correlation with Industry Switches:** -0.007
- **Correlation with Technical Degree:** -0.005
- **Findings:** There is no statistically significant correlation between the trust flags and nonlinear career indicators.
- **Mitigation Applied:** The YOE-tenure check employs a minimum deviation threshold of 1.5 years and a scaled penalty, which successfully tolerates typical multi-year gap patterns and prevents bias against non-traditional trajectories.
