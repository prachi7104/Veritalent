"use client";

import React, { useState, useEffect, useRef } from "react";
import { useRouter } from "next/navigation";
import { useSearchState } from "@/lib/SearchStateProvider";
import { rerankByJD } from "@/lib/api";
import { CandidateCard } from "@/components/CandidateCard";
import { ScenarioExplorer } from "@/components/ScenarioExplorer";
import { Settings2 } from "lucide-react";

export default function Discovery() {
  const router = useRouter();
  const { searchResponse, setSearchResponse, lastQuery, setLastQuery, setSearchLatency } = useSearchState();
  const [jdText, setJdText] = useState(lastQuery);
  const [isEditing, setIsEditing] = useState(false);
  const [explorerOpen, setExplorerOpen] = useState(false);
  
  // Multi-select for Compare (2-4 candidates)
  const [selectedIds, setSelectedIds] = useState<string[]>([]);

  // Debounce ref
  const debounceRef = useRef<NodeJS.Timeout | null>(null);

  useEffect(() => {
    // If no context available, we are in a cold state
    if (!searchResponse) {
      return;
    }
  }, [searchResponse]);

  if (!searchResponse) {
    return (
      <div style={{ maxWidth: "800px", margin: "64px auto", textAlign: "center" }}>
        <h2 style={{ marginBottom: "16px", color: "var(--text-primary)" }}>No active search</h2>
        <button
          onClick={() => router.push("/")}
          style={{
            padding: "10px 16px",
            backgroundColor: "var(--bg-accent)",
            color: "var(--text-accent)",
            border: "none",
            borderRadius: "var(--radius)",
            cursor: "pointer",
            fontWeight: 600
          }}
        >
          Return to Home
        </button>
      </div>
    );
  }

  const { funnel_stats, jd_decomposition, candidates } = searchResponse;

  const handleJdChange = (e: React.ChangeEvent<HTMLTextAreaElement>) => {
    const newText = e.target.value;
    setJdText(newText);
    
    if (debounceRef.current) clearTimeout(debounceRef.current);
    
    debounceRef.current = setTimeout(async () => {
      const startTime = performance.now();
      try {
        const res = await rerankByJD({
          session_id: searchResponse.session_id,
          updated_jd_text: newText,
          top_k: searchResponse.candidates.length || 100,
        });
        const endTime = performance.now();
        setSearchLatency(Math.round(endTime - startTime));
        setSearchResponse(res);
        setLastQuery(newText);
      } catch (err) {
        // Silently catch errors without clearing the list as per spec
        console.error("Live rerank failed:", err);
      }
    }, 600);
  };

  const handleCheckChange = (candidateId: string, checked: boolean) => {
    if (checked) {
      if (selectedIds.length < 4) {
        setSelectedIds([...selectedIds, candidateId]);
      }
    } else {
      setSelectedIds(selectedIds.filter(id => id !== candidateId));
    }
  };

  const handleCompareClick = () => {
    if (selectedIds.length >= 2 && selectedIds.length <= 4) {
      router.push(`/compare?ids=${selectedIds.join(",")}`);
    }
  };

  return (
    <div style={{ maxWidth: "900px", margin: "0 auto", padding: "32px 16px", paddingBottom: "100px" }}>
      {/* 1. Funnel Strip */}
      <div style={{ fontSize: "0.85rem", color: "var(--text-muted)", marginBottom: "24px" }}>
        {funnel_stats.total_pool.toLocaleString()} total &rarr; {funnel_stats.title_relevant.toLocaleString()} title-relevant &rarr; {funnel_stats.retrieved} retrieved &rarr; {funnel_stats.shown} shown
      </div>

      {/* 2. JD Chips */}
      <div style={{ display: "flex", flexWrap: "wrap", gap: "8px", marginBottom: "24px" }}>
        {jd_decomposition.must_haves.map((mh, idx) => (
          <span key={`mh-${idx}`} style={{ padding: "4px 8px", backgroundColor: "var(--bg-accent)", color: "var(--text-accent)", borderRadius: "var(--radius)", fontSize: "0.85rem", fontWeight: 500 }}>
            {mh}
          </span>
        ))}
        {jd_decomposition.nice_to_haves.map((nth, idx) => (
          <span key={`nth-${idx}`} style={{ padding: "4px 8px", backgroundColor: "var(--surface-1)", color: "var(--text-secondary)", borderRadius: "var(--radius)", fontSize: "0.85rem" }}>
            {nth}
          </span>
        ))}
        {jd_decomposition.fallback_used && (
          <span style={{ padding: "4px 8px", backgroundColor: "var(--bg-warning)", color: "var(--text-warning)", borderRadius: "var(--radius)", fontSize: "0.85rem", fontWeight: 600 }}>
            keyword fallback used
          </span>
        )}
      </div>

      {/* 3. Editable JD */}
      <div style={{ marginBottom: "32px", padding: "16px", backgroundColor: "var(--surface-0)", borderRadius: "var(--radius)", border: "1px solid var(--border)" }}>
        <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: "8px" }}>
          <h3 style={{ fontSize: "1rem", margin: 0 }}>Job Description</h3>
          <button 
            onClick={() => setIsEditing(!isEditing)}
            style={{ background: "none", border: "none", color: "var(--text-accent)", cursor: "pointer", fontSize: "0.9rem" }}
          >
            {isEditing ? "Collapse" : "Edit"}
          </button>
        </div>
        
        {isEditing ? (
          <>
            <textarea
              value={jdText}
              onChange={handleJdChange}
              style={{
                width: "100%",
                minHeight: "100px",
                padding: "8px",
                borderRadius: "4px",
                border: "1px solid var(--border-strong)",
                fontFamily: "inherit",
                fontSize: "0.95rem",
                resize: "vertical"
              }}
            />
            <div style={{ fontSize: "0.8rem", color: "var(--text-muted)", marginTop: "8px" }}>
              Editing re-ranks live. The explanations below may lag a moment behind your latest edit by design, not by bug.
            </div>
          </>
        ) : (
          <div style={{ fontSize: "0.95rem", color: "var(--text-secondary)", whiteSpace: "nowrap", overflow: "hidden", textOverflow: "ellipsis" }}>
            {jdText}
          </div>
        )}
      </div>

      {/* 4. Candidate List */}
      <div>
        {candidates.map(cand => (
          <CandidateCard
            key={cand.candidate_id}
            candidate={cand}
            showCheckbox={true}
            checked={selectedIds.includes(cand.candidate_id)}
            onCheckChange={(checked) => handleCheckChange(cand.candidate_id, checked)}
            disabledCheckbox={selectedIds.length >= 4}
            showShortlistToggle={true}
          />
        ))}
      </div>

      {/* 5. Scenario Explorer */}
      {!explorerOpen ? (
        <div style={{ marginTop: "32px", borderTop: "1px solid var(--border)", paddingTop: "16px" }}>
          <button 
            onClick={() => setExplorerOpen(true)}
            style={{
              display: "flex",
              alignItems: "center",
              gap: "8px",
              background: "none",
              border: "none",
              color: "var(--text-accent)",
              fontWeight: 500,
              cursor: "pointer",
              padding: "8px",
              borderRadius: "var(--radius)",
            }}
          >
            <Settings2 size={18} />
            <span>Explore weight scenarios</span>
          </button>
        </div>
      ) : (
        <ScenarioExplorer isOpen={explorerOpen} onClose={() => setExplorerOpen(false)} />
      )}

      {/* Compare Floating Bar */}
      {selectedIds.length >= 2 && (
        <div
          style={{
            position: "fixed",
            bottom: "24px",
            left: "50%",
            transform: "translateX(-50%)",
            backgroundColor: "var(--surface-0)",
            border: "1px solid var(--border-strong)",
            boxShadow: "0 10px 25px -5px rgba(0, 0, 0, 0.1), 0 8px 10px -6px rgba(0, 0, 0, 0.1)",
            padding: "12px 24px",
            borderRadius: "999px",
            display: "flex",
            alignItems: "center",
            gap: "16px",
            zIndex: 100
          }}
        >
          <span style={{ fontSize: "0.95rem", fontWeight: 500 }}>
            {selectedIds.length} candidate{selectedIds.length !== 1 ? "s" : ""} selected
          </span>
          <button
            onClick={handleCompareClick}
            disabled={selectedIds.length < 2 || selectedIds.length > 4}
            style={{
              padding: "8px 16px",
              backgroundColor: "var(--bg-accent)",
              color: "var(--text-accent)",
              border: "none",
              borderRadius: "999px",
              fontWeight: 600,
              cursor: "pointer",
              fontSize: "0.9rem"
            }}
          >
            Compare selected
          </button>
        </div>
      )}
    </div>
  );
}
