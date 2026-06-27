"use client";

import React, { useEffect, useState, use } from "react";
import { getCandidate } from "@/lib/api";
import { useSearchState } from "@/lib/SearchStateProvider";
import { FeatureBar } from "@/components/FeatureBar";
import { TrustBadge } from "@/components/TrustBadge";
import { FINGERPRINT_CAVEAT } from "@/lib/constants";
import { Star } from "lucide-react";
import type { CandidateDetailResponse, SkillGap, CandidateCardResponse } from "@/lib/types";

export default function CandidateDetail({ params }: { params: Promise<{ id: string }> }) {
  const resolvedParams = use(params);
  const candidateId = decodeURIComponent(resolvedParams.id);
  const [candidate, setCandidate] = useState<CandidateDetailResponse | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [checksExpanded, setChecksExpanded] = useState(false);
  
  const { searchResponse, shortlistedCandidates, toggleShortlist } = useSearchState();

  // Cold link explicit lookup mechanism
  let rank: number | null = null;
  let skillGap: SkillGap | null = null;
  
  if (searchResponse && searchResponse.candidates) {
    const match = searchResponse.candidates.find(c => c.candidate_id === candidateId);
    if (match) {
      rank = match.rank;
      skillGap = match.skill_gap;
    }
  }

  useEffect(() => {
    async function fetchDetail() {
      try {
        const res = await getCandidate(candidateId);
        setCandidate(res);
      } catch (err: unknown) {
        setError((err as Error).message || "Failed to load candidate.");
      }
    }
    fetchDetail();
  }, [candidateId]);

  if (error) {
    return <div style={{ color: "var(--text-danger)", padding: "32px", textAlign: "center" }}>{error}</div>;
  }

  if (!candidate) {
    return <div style={{ padding: "32px", textAlign: "center", color: "var(--text-muted)" }}>Loading profile...</div>;
  }

  const { profile, shap_attribution, trust_breakdown } = candidate;

  // Synthesize CandidateCardResponse for shortlist toggle
  const cardMatch = searchResponse?.candidates.find(c => c.candidate_id === candidateId);
  const candidateCardObj: CandidateCardResponse = cardMatch || {
    candidate_id: candidate.candidate_id,
    rank: rank || 0,
    score: 0,
    current_title: profile.current_title,
    current_company: profile.current_company,
    years_of_experience: profile.years_of_experience,
    location: profile.location,
    top_features: [],
    trust_score: trust_breakdown.composite_score,
    trust_level: trust_breakdown.level,
    fingerprint_holder: candidate.fingerprint_holder,
    narrative: candidate.narrative,
    narrative_is_llm: candidate.narrative_is_llm,
    fallback_used: false,
    skill_gap: skillGap || { missing_deep_ir_skills: [], matched_deep_ir_skills: [], gap_to_next_tier: null }
  };

  const isShortlisted = shortlistedCandidates.some((c) => c.candidate_id === candidateId);

  // Find max absolute value for SHAP scaling
  const maxShap = shap_attribution.top_features.reduce(
    (max, feat) => Math.max(max, Math.abs(feat.shap_contribution)), 
    0
  );

  return (
    <div style={{ maxWidth: "800px", margin: "0 auto", padding: "32px 16px" }}>
      {/* 1. Header */}
      <div style={{ display: "flex", justifyContent: "space-between", alignItems: "flex-start", marginBottom: "32px" }}>
        <div style={{ display: "flex", alignItems: "flex-start", gap: "16px" }}>
          <button
            onClick={() => toggleShortlist(candidateCardObj)}
            style={{
              background: "none",
              border: "none",
              cursor: "pointer",
              padding: "8px",
              display: "flex",
              alignItems: "center",
              justifyContent: "center",
              color: isShortlisted ? "#eab308" : "var(--text-muted)",
              marginTop: "4px"
            }}
          >
            <Star size={24} fill={isShortlisted ? "#eab308" : "none"} />
          </button>
          <div>
            <h1 style={{ fontSize: "1.5rem", fontWeight: 700, margin: "0 0 8px 0", color: "var(--text-primary)" }}>
              {profile.current_title}
            </h1>
            <div style={{ color: "var(--text-secondary)", fontSize: "1.1rem" }}>
              {profile.current_company} &middot; {profile.years_of_experience} yrs &middot; {profile.location}
            </div>
          </div>
        </div>
        {rank !== null && (
          <div style={{ fontSize: "1.25rem", fontWeight: 600, color: "var(--text-muted)", backgroundColor: "var(--surface-0)", padding: "8px 16px", borderRadius: "var(--radius)", border: "1px solid var(--border)" }}>
            Rank #{rank}
          </div>
        )}
      </div>

      {/* 2. Fingerprint Badge */}
      {candidate.fingerprint_holder && (
        <div style={{ marginBottom: "32px", padding: "16px", backgroundColor: "var(--bg-accent)", borderRadius: "var(--radius)", border: "1px solid var(--border-strong)" }}>
          <div style={{ color: "var(--text-accent)", fontWeight: 700, marginBottom: "8px" }}>Rare skill pattern</div>
          <div style={{ fontSize: "0.85rem", color: "var(--text-secondary)", lineHeight: 1.5 }}>
            {FINGERPRINT_CAVEAT}
          </div>
        </div>
      )}

      {/* 3. Why this rank */}
      <div style={{ marginBottom: "48px" }}>
        <h2 style={{ fontSize: "1.2rem", fontWeight: 600, marginBottom: "16px", borderBottom: "1px solid var(--border)", paddingBottom: "8px" }}>
          Why this rank
        </h2>
        {shap_attribution.top_features.length === 0 ? (
          <div style={{ color: "var(--text-muted)", fontStyle: "italic" }}>No distinct feature attributions available.</div>
        ) : (
          <div>
            {shap_attribution.top_features.map((feat, idx) => (
              <FeatureBar key={idx} feature={feat} maxAbsValue={maxShap || 1} />
            ))}
          </div>
        )}
      </div>

      {/* 4. Skill gap (if available via context props) */}
      {skillGap && (
        <div style={{ marginBottom: "48px" }}>
          <h2 style={{ fontSize: "1.2rem", fontWeight: 600, marginBottom: "16px", borderBottom: "1px solid var(--border)", paddingBottom: "8px" }}>
            Skill Evaluation
          </h2>
          <div style={{ display: "flex", flexWrap: "wrap", gap: "8px", marginBottom: "16px" }}>
            {skillGap.matched_deep_ir_skills.map((s, idx) => (
              <span key={`matched-${idx}`} style={{ padding: "4px 8px", backgroundColor: "var(--bg-success)", color: "var(--text-success)", borderRadius: "var(--radius)", fontSize: "0.85rem", fontWeight: 500 }}>
                {s}
              </span>
            ))}
            {skillGap.missing_deep_ir_skills.map((s, idx) => (
              <span key={`missing-${idx}`} style={{ padding: "4px 8px", backgroundColor: "var(--surface-1)", color: "var(--text-secondary)", borderRadius: "var(--radius)", fontSize: "0.85rem" }}>
                {s}
              </span>
            ))}
          </div>
          {skillGap.gap_to_next_tier !== null && (
            <div style={{ fontSize: "0.9rem", color: "var(--text-muted)" }}>
              {skillGap.gap_to_next_tier} &mdash; approximate
            </div>
          )}
        </div>
      )}

      {/* 5. Trust */}
      <div style={{ marginBottom: "32px", padding: "16px", backgroundColor: "var(--surface-0)", borderRadius: "var(--radius)", border: "1px solid var(--border)" }}>
        <div style={{ display: "flex", alignItems: "center", gap: "16px", marginBottom: "12px" }}>
          <h2 style={{ fontSize: "1.2rem", fontWeight: 600, margin: 0 }}>Trust Validation</h2>
          <TrustBadge level={trust_breakdown.level} score={trust_breakdown.composite_score} />
        </div>
        
        <div style={{ fontSize: "0.95rem", color: "var(--text-secondary)", marginBottom: "16px", lineHeight: 1.5 }}>
          {trust_breakdown.caveat}
        </div>
        
        <button 
          onClick={() => setChecksExpanded(!checksExpanded)}
          style={{ background: "none", border: "none", color: "var(--text-accent)", cursor: "pointer", fontWeight: 600, padding: 0 }}
        >
          {checksExpanded ? "Hide check breakdown" : "View check breakdown"}
        </button>
        
        {checksExpanded && (
          <div style={{ marginTop: "16px", display: "flex", flexDirection: "column", gap: "12px" }}>
            {Object.entries(trust_breakdown.checks).map(([key, check]) => {
              const isFlagged = check.flagged;
              return (
                <div key={key} style={{ padding: "12px", borderLeft: `3px solid ${isFlagged ? "var(--text-danger)" : "var(--text-success)"}`, backgroundColor: "var(--surface-1)" }}>
                  <div style={{ fontWeight: 600, color: "var(--text-primary)", marginBottom: "4px" }}>
                    {isFlagged ? "⚠️ Flagged" : "✓ Clear"}: {key.replace(/_/g, " ")}
                  </div>
                  <div style={{ fontSize: "0.9rem", color: "var(--text-secondary)" }}>
                    {check.explanation}
                  </div>
                </div>
              );
            })}
          </div>
        )}
      </div>
    </div>
  );
}
