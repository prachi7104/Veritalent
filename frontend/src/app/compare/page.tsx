"use client";

import React, { useEffect, useState, use } from "react";
import { useRouter } from "next/navigation";
import { compareCandidates } from "@/lib/api";
import { useSearchState } from "@/lib/SearchStateProvider";
import { TrustBadge } from "@/components/TrustBadge";
import type { CompareResponse } from "@/lib/types";

export default function Compare({ searchParams }: { searchParams: Promise<{ ids?: string }> }) {
  const router = useRouter();
  const resolvedSearchParams = use(searchParams);
  const idsString = resolvedSearchParams.ids || "";
  const candidateIds = idsString ? idsString.split(",") : [];

  const [compareData, setCompareData] = useState<CompareResponse | null>(null);
  const [loading, setLoading] = useState(false);
  const [errorMsg, setErrorMsg] = useState<string | null>(null);

  const { searchResponse } = useSearchState();

  // Floor Check (and Ceiling Check)
  const isFloorValid = candidateIds.length >= 2;
  const isCeilingValid = candidateIds.length <= 4;

  useEffect(() => {
    if (!isFloorValid || !isCeilingValid) {
      return;
    }

    async function fetchComparison() {
      setLoading(true);
      setErrorMsg(null);
      try {
        const res = await compareCandidates({ candidate_ids: candidateIds });
        setCompareData(res);
      } catch (err: unknown) {
        setErrorMsg((err as Error).message || "Failed to load comparison data.");
      } finally {
        setLoading(false);
      }
    }

    fetchComparison();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [idsString]);

  if (!isFloorValid) {
    return (
      <div style={{ maxWidth: "600px", margin: "64px auto", textAlign: "center" }}>
        <h2 style={{ marginBottom: "16px", color: "var(--text-primary)" }}>Comparison requires at least 2 candidates</h2>
        <button
          onClick={() => router.push("/discovery")}
          style={{
            padding: "10px 16px",
            backgroundColor: "var(--bg-accent)",
            color: "var(--text-accent)",
            border: "none",
            borderRadius: "var(--radius)",
            cursor: "pointer",
            fontWeight: 600,
          }}
        >
          Return to Discovery
        </button>
      </div>
    );
  }

  if (!isCeilingValid) {
    return (
      <div style={{ maxWidth: "600px", margin: "64px auto", textAlign: "center" }}>
        <h2 style={{ marginBottom: "16px", color: "var(--text-primary)" }}>Compare supports 2–4 candidates maximum.</h2>
        <button
          onClick={() => router.push("/discovery")}
          style={{
            padding: "10px 16px",
            backgroundColor: "var(--bg-accent)",
            color: "var(--text-accent)",
            border: "none",
            borderRadius: "var(--radius)",
            cursor: "pointer",
            fontWeight: 600,
          }}
        >
          Return to Discovery
        </button>
      </div>
    );
  }

  if (errorMsg) {
    return (
      <div style={{ maxWidth: "600px", margin: "64px auto", textAlign: "center", color: "var(--text-danger)" }}>
        <p>{errorMsg}</p>
        <button
          onClick={() => router.push("/discovery")}
          style={{
            marginTop: "16px",
            padding: "10px 16px",
            backgroundColor: "var(--bg-accent)",
            color: "var(--text-accent)",
            border: "none",
            borderRadius: "var(--radius)",
            cursor: "pointer",
            fontWeight: 600,
          }}
        >
          Return to Discovery
        </button>
      </div>
    );
  }

  if (loading || !compareData) {
    return <div style={{ padding: "64px", textAlign: "center", color: "var(--text-muted)" }}>Loading comparison...</div>;
  }

  const { candidates, comparison_matrix } = compareData;

  return (
    <div style={{ maxWidth: "1200px", margin: "0 auto", padding: "32px 16px" }}>
      <h1 style={{ fontSize: "1.5rem", marginBottom: "24px" }}>Candidate Comparison</h1>
      
      <div style={{ overflowX: "auto" }}>
        <table style={{ width: "100%", borderCollapse: "collapse", minWidth: "600px" }}>
          <thead>
            <tr style={{ borderBottom: "2px solid var(--border-strong)" }}>
              {/* Feature Label Column */}
              <th style={{ textAlign: "left", padding: "16px", width: "200px", color: "var(--text-secondary)" }}>
                Metric
              </th>
              
              {/* Candidate Columns */}
              {candidates.map((cand) => {
                // Graceful lookup of rank from context
                let rank: number | null = null;
                if (searchResponse && searchResponse.candidates) {
                  const match = searchResponse.candidates.find(c => c.candidate_id === cand.candidate_id);
                  if (match) rank = match.rank;
                }
                
                return (
                  <th key={cand.candidate_id} style={{ textAlign: "left", padding: "16px", verticalAlign: "top" }}>
                    <div style={{ fontWeight: 600, fontSize: "1.05rem", color: "var(--text-primary)" }}>
                      {cand.profile.current_title}
                    </div>
                    <div style={{ fontSize: "0.85rem", color: "var(--text-muted)", marginTop: "4px" }}>
                      {cand.profile.current_company}
                    </div>
                    {rank !== null && (
                      <div style={{ display: "inline-block", fontSize: "0.8rem", fontWeight: 700, backgroundColor: "var(--surface-1)", padding: "2px 6px", borderRadius: "4px", marginTop: "8px" }}>
                        Rank #{rank}
                      </div>
                    )}
                  </th>
                );
              })}
            </tr>
          </thead>
          
          <tbody>
            {/* Trust Level Row */}
            <tr style={{ borderBottom: "1px solid var(--border)" }}>
              <td style={{ padding: "16px", fontWeight: 500, color: "var(--text-secondary)" }}>Trust Validation</td>
              {candidates.map((cand) => (
                <td key={cand.candidate_id} style={{ padding: "16px" }}>
                  <TrustBadge level={cand.trust_breakdown.level} />
                </td>
              ))}
            </tr>
            
            {/* Matrix Features dynamically resolved */}
            {comparison_matrix.features.map((feat) => (
              <tr key={feat} style={{ borderBottom: "1px solid var(--border)" }}>
                <td style={{ padding: "16px", fontFamily: "var(--font-mono)", fontSize: "0.9rem", color: "var(--text-secondary)" }}>
                  {feat}
                </td>
                {candidates.map((cand) => {
                  const value = comparison_matrix.values[feat]?.[cand.candidate_id];
                  const delta = comparison_matrix.deltas[feat]?.[cand.candidate_id];
                  
                  const isValMissing = value === undefined;
                  const isDeltaMissing = delta === undefined;

                  return (
                    <td key={cand.candidate_id} style={{ padding: "16px", fontSize: "0.95rem" }}>
                      {isValMissing ? (
                        <span style={{ color: "var(--text-muted)" }}>—</span>
                      ) : (
                        <span>
                          {value.toFixed(2)}{" "}
                          {!isDeltaMissing && (
                            <span style={{ color: "var(--text-secondary)", fontSize: "0.85rem", marginLeft: "4px" }}>
                              ({delta >= 0 ? `+${delta.toFixed(2)}` : delta.toFixed(2)})
                            </span>
                          )}
                        </span>
                      )}
                    </td>
                  );
                })}
              </tr>
            ))}
            
            {/* Footer / Link to Profile Row */}
            <tr>
              <td style={{ padding: "16px" }}></td>
              {candidates.map((cand) => (
                <td key={cand.candidate_id} style={{ padding: "16px" }}>
                  <button
                    onClick={() => router.push(`/candidate/${encodeURIComponent(cand.candidate_id)}`)}
                    style={{
                      padding: "8px 12px",
                      backgroundColor: "var(--surface-1)",
                      border: "1px solid var(--border)",
                      borderRadius: "var(--radius)",
                      cursor: "pointer",
                      fontWeight: 600,
                      fontSize: "0.85rem",
                      width: "100%",
                      textAlign: "center"
                    }}
                  >
                    View profile
                  </button>
                </td>
              ))}
            </tr>
          </tbody>
        </table>
      </div>
    </div>
  );
}
