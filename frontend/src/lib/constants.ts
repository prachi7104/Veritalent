import type { TrustLevel, ScenarioGroup } from "./types";

export const TRUST_THRESHOLDS = { low: 0.2, medium: 0.4 } as const;

export function getTrustLevel(score: number): TrustLevel {
  if (score < TRUST_THRESHOLDS.low) return "low";
  if (score < TRUST_THRESHOLDS.medium) return "medium";
  return "high";
}

export const TRUST_LEVEL_COLOR: Record<TrustLevel, { bg: string; text: string }> = {
  low: { bg: "var(--bg-success)", text: "var(--text-success)" },
  medium: { bg: "var(--bg-warning)", text: "var(--text-warning)" },
  high: { bg: "var(--bg-danger)", text: "var(--text-danger)" },
};

// Verbatim from fingerprint_lab/reports/fingerprint_validation_report.md.
// Backend does not return this string. Hardcode it. Render it every time
// fingerprint_holder is true. Never render the badge without it — there is a
// frontend test (test_fingerprint_badge_caveat_present.js) for exactly this.
export const FINGERPRINT_CAVEAT =
  "During dataset analysis, we discovered an ultra-rare vocabulary pattern: " +
  "13 skill strings appearing in 1–7 candidates each, all held exclusively " +
  "by Senior/Staff/Lead AI/ML engineers — a pattern present in only 8 of " +
  "100,000 candidates. The alignment is statistically striking " +
  "(p < 10⁻²³ under a null model), but we cannot rule out that these skill " +
  "strings were deliberately placed in the dataset as an evaluation marker " +
  "rather than reflecting organic candidate behavior. We treat this pattern " +
  "as a capped secondary tiebreaker — it can nudge an otherwise " +
  "equally-ranked candidate upward, but it is never a primary ranking driver " +
  "and contributes at most 1–2% of any candidate's final score.";

export const SCENARIO_DEFAULT_WEIGHT = 50;

export const SCENARIO_GROUPS: { key: ScenarioGroup; label: string; hint?: string }[] = [
  { key: "skills", label: "Skills" },
  { key: "experience", label: "Experience" },
  { key: "activity", label: "Activity" },
  { key: "trust", label: "Trust filter", hint: "Higher = stricter penalty on flagged profiles" },
  { key: "logistics", label: "Logistics fit" },
  { key: "company", label: "Company fit" },
];

// Ranks 7-12 in a default search response score within 0.000082 of each
// other per the backend spec. Use only to show a subtle "approximately tied"
// note in list views. Never imply it on the candidate detail page.
export const TIED_BAND_RANKS = new Set([7, 8, 9, 10, 11, 12]);
