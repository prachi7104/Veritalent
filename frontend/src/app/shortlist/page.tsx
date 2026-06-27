"use client";

import React from "react";
import { useSearchState } from "@/lib/SearchStateProvider";
import { Star, Trash2 } from "lucide-react";
import { useRouter } from "next/navigation";

export default function Shortlist() {
  const router = useRouter();
  const { shortlistedCandidates, toggleShortlist } = useSearchState();

  const handleExport = () => {
    // Stub the export action
    alert("Export feature stubbed! shortlists: " + JSON.stringify(shortlistedCandidates.map(c => c.candidate_id)));
  };

  return (
    <div style={{ maxWidth: "800px", margin: "0 auto", padding: "32px 16px" }}>
      {/* Header */}
      <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: "32px", borderBottom: "1px solid var(--border)", paddingBottom: "16px" }}>
        <h1 style={{ fontSize: "1.5rem", fontWeight: 700 }}>
          {shortlistedCandidates.length} Shortlisted
        </h1>
        <button
          onClick={handleExport}
          disabled={shortlistedCandidates.length === 0}
          style={{
            padding: "8px 16px",
            backgroundColor: shortlistedCandidates.length === 0 ? "var(--surface-1)" : "var(--bg-accent)",
            color: shortlistedCandidates.length === 0 ? "var(--text-muted)" : "var(--text-accent)",
            border: "none",
            borderRadius: "var(--radius)",
            cursor: shortlistedCandidates.length === 0 ? "default" : "pointer",
            fontWeight: 600,
            fontSize: "0.9rem"
          }}
        >
          Export shortlist
        </button>
      </div>

      {/* List */}
      {shortlistedCandidates.length === 0 ? (
        <div style={{ textAlign: "center", padding: "48px 16px", color: "var(--text-muted)" }}>
          <Star size={48} style={{ marginBottom: "16px", opacity: 0.5 }} />
          <p style={{ fontSize: "1.1rem", marginBottom: "16px" }}>Your shortlist is empty.</p>
          <button
            onClick={() => router.push("/")}
            style={{
              padding: "8px 16px",
              backgroundColor: "var(--surface-1)",
              border: "1px solid var(--border)",
              borderRadius: "var(--radius)",
              cursor: "pointer",
              fontWeight: 500
            }}
          >
            Find Candidates
          </button>
        </div>
      ) : (
        <div style={{ display: "flex", flexDirection: "column", gap: "12px" }}>
          {shortlistedCandidates.map((cand) => (
            <div
              key={cand.candidate_id}
              style={{
                display: "flex",
                alignItems: "center",
                padding: "16px",
                border: "1px solid var(--border)",
                borderRadius: "var(--radius)",
                backgroundColor: "var(--surface-0)",
              }}
            >
              {/* Actual Search Rank (non-sequentialized) */}
              <div style={{ flex: "0 0 50px", fontSize: "1.2rem", fontWeight: "bold", color: "var(--text-muted)" }}>
                #{cand.rank}
              </div>

              <div style={{ flex: 1, paddingLeft: "16px" }}>
                <div 
                  onClick={() => router.push(`/candidate/${encodeURIComponent(cand.candidate_id)}`)}
                  style={{ fontWeight: 600, fontSize: "1.05rem", color: "var(--text-primary)", cursor: "pointer", display: "inline-block" }}
                >
                  {cand.current_title}
                </div>
                <div style={{ color: "var(--text-secondary)", fontSize: "0.9rem" }}>
                  {cand.current_company} · {cand.years_of_experience} yrs · {cand.location}
                </div>
              </div>

              {/* Remove Star Button */}
              <button
                onClick={() => toggleShortlist(cand)}
                style={{
                  background: "none",
                  border: "none",
                  cursor: "pointer",
                  color: "var(--text-danger)",
                  padding: "8px",
                  display: "flex",
                  alignItems: "center",
                  justifyContent: "center"
                }}
              >
                <Trash2 size={20} />
              </button>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}
