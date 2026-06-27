import React, { useState, useRef } from "react";
import { useRouter } from "next/navigation";
import { useSearchState } from "@/lib/SearchStateProvider";
import { scenarioRerank, ApiError } from "@/lib/api";
import { SCENARIO_GROUPS } from "@/lib/constants";
import type { ScenarioGroup, ScenarioCandidate } from "@/lib/types";

interface ScenarioExplorerProps {
  isOpen: boolean;
  onClose: () => void;
}

export function ScenarioExplorer({ isOpen, onClose }: ScenarioExplorerProps) {
  const router = useRouter();
  const { searchResponse, setScenarioInteraction } = useSearchState();

  const [mode, setMode] = useState<"default" | "custom">("default");
  const [weights, setWeights] = useState<Record<ScenarioGroup, number>>({
    skills: 50,
    experience: 50,
    activity: 50,
    trust: 50,
    logistics: 50,
    company: 50,
  });
  
  const [reRankedList, setReRankedList] = useState<ScenarioCandidate[]>([]);
  const [loading, setLoading] = useState(false);
  const [errorMsg, setErrorMsg] = useState<string | null>(null);
  
  const debounceRef = useRef<NodeJS.Timeout | null>(null);

  const fetchRerank = async (
    currentWeights: Record<ScenarioGroup, number>,
    originatingGroup?: ScenarioGroup
  ) => {
    if (!searchResponse) return;
    setLoading(true);
    setErrorMsg(null);
    try {
      const res = await scenarioRerank({
        session_id: searchResponse.session_id,
        weight_overrides: currentWeights,
      });
      setReRankedList(res.re_ranked);

      // Only record interaction if triggered by a slider onChange event
      if (originatingGroup) {
        // Find maximum absolute delta
        let maxDelta = 0;
        res.re_ranked.forEach((c) => {
          const absDelta = Math.abs(c.rank_delta);
          if (absDelta > maxDelta) {
            maxDelta = absDelta;
          }
        });

        // Only record if some rank movement occurred
        if (maxDelta > 0) {
          // Filter candidates with maximum delta
          const candidatesWithMaxDelta = res.re_ranked.filter(
            (c) => Math.abs(c.rank_delta) === maxDelta
          );

          // Tie-break: select the one with the lowest new_rank (meaning highest rank e.g. #1 over #5)
          let winner = candidatesWithMaxDelta[0];
          candidatesWithMaxDelta.forEach((c) => {
            if (c.new_rank < winner.new_rank) {
              winner = c;
            }
          });

          // Retrieve candidate title
          const match = searchResponse.candidates.find(
            (c) => c.candidate_id === winner.candidate_id
          );
          const candidateTitle = match?.current_title || "a candidate";

          // Save the interaction details
          setScenarioInteraction({
            group: originatingGroup,
            value: currentWeights[originatingGroup],
            originalRank: winner.original_rank,
            newRank: winner.new_rank,
            candidateTitle,
          });
        }
      }
    } catch (err: unknown) {
      if (err instanceof ApiError && err.status === 404) {
        setErrorMsg("Session expired or not found. Please re-submit your search.");
      } else {
        setErrorMsg((err as Error).message || "Failed to execute scenario re-rank.");
      }
    } finally {
      setLoading(false);
    }
  };

  // Toggle Mode Handler
  const handleModeChange = (newMode: "default" | "custom") => {
    setMode(newMode);
    if (newMode === "custom") {
      // Transitioning Default -> Custom triggers one immediate call
      fetchRerank(weights);
    } else {
      // Custom -> Default discards custom list and makes no API calls
      setReRankedList([]);
    }
  };

  const handleSliderChange = (group: ScenarioGroup, val: number) => {
    const nextWeights = { ...weights, [group]: val };
    setWeights(nextWeights);

    if (debounceRef.current) clearTimeout(debounceRef.current);

    debounceRef.current = setTimeout(() => {
      fetchRerank(nextWeights, group);
    }, 250);
  };

  if (!isOpen || !searchResponse) return null;

  return (
    <div style={{ marginTop: "24px", padding: "24px", border: "1px solid var(--border-strong)", borderRadius: "var(--radius)", backgroundColor: "var(--surface-0)" }}>
      <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: "20px" }}>
        <h3 style={{ margin: 0, fontSize: "1.2rem" }}>Weight Scenarios</h3>
        <button onClick={onClose} style={{ background: "none", border: "none", color: "var(--text-accent)", cursor: "pointer", fontWeight: 600 }}>
          Close Panel
        </button>
      </div>

      {/* Mode Selection Cards */}
      <div style={{ display: "flex", gap: "16px", marginBottom: "24px" }}>
        <div
          onClick={() => handleModeChange("default")}
          style={{
            flex: 1,
            padding: "16px",
            border: `2px solid ${mode === "default" ? "var(--border-strong)" : "var(--border)"}`,
            borderRadius: "var(--radius)",
            backgroundColor: mode === "default" ? "var(--surface-1)" : "var(--surface-0)",
            cursor: "pointer",
          }}
        >
          <div style={{ fontWeight: 600, marginBottom: "4px" }}>Default ranking</div>
          <div style={{ fontSize: "0.85rem", color: "var(--text-secondary)" }}>
            The primary model&apos;s ranking. Weights stay fixed.
          </div>
        </div>

        <div
          onClick={() => handleModeChange("custom")}
          style={{
            flex: 1,
            padding: "16px",
            border: `2px solid ${mode === "custom" ? "var(--border-strong)" : "var(--border)"}`,
            borderRadius: "var(--radius)",
            backgroundColor: mode === "custom" ? "var(--surface-1)" : "var(--surface-0)",
            cursor: "pointer",
          }}
        >
          <div style={{ fontWeight: 600, marginBottom: "4px" }}>Custom weights</div>
          <div style={{ fontSize: "0.85rem", color: "var(--text-secondary)" }}>
            Adjust the sliders to explore alternative rankings.
          </div>
        </div>
      </div>

      {errorMsg ? (
        <div style={{ color: "var(--text-danger)", marginBottom: "16px", textAlign: "center" }}>
          <p>{errorMsg}</p>
          <button
            onClick={() => router.push("/")}
            style={{
              marginTop: "8px",
              padding: "8px 16px",
              backgroundColor: "var(--bg-accent)",
              color: "var(--text-accent)",
              border: "none",
              borderRadius: "var(--radius)",
              cursor: "pointer",
              fontWeight: 600,
            }}
          >
            Re-submit search (Go to Home)
          </button>
        </div>
      ) : (
        <fieldset disabled={mode === "default"} style={{ border: "none", padding: 0, margin: 0 }}>
          <div style={{ display: "grid", gridTemplateColumns: "repeat(2, 1fr)", gap: "16px", marginBottom: "24px" }}>
            {SCENARIO_GROUPS.map((g) => (
              <div key={g.key} style={{ display: "flex", flexDirection: "column", gap: "4px" }}>
                <div style={{ display: "flex", justifyContent: "space-between", fontSize: "0.95rem", fontWeight: 500 }}>
                  <label htmlFor={`slider-${g.key}`}>{g.label}</label>
                  <span>{weights[g.key]}</span>
                </div>
                <input
                  id={`slider-${g.key}`}
                  type="range"
                  min="0"
                  max="100"
                  value={weights[g.key]}
                  onChange={(e) => handleSliderChange(g.key, parseInt(e.target.value))}
                  disabled={mode === "default"}
                  style={{ width: "100%", cursor: mode === "default" ? "default" : "pointer" }}
                />
                {g.hint && (
                  <span style={{ fontSize: "0.8rem", color: "var(--text-muted)" }}>
                    {g.hint}
                  </span>
                )}
              </div>
            ))}
          </div>
        </fieldset>
      )}

      {/* Result Rows */}
      {mode === "custom" && reRankedList.length > 0 && !loading && !errorMsg && (
        <div style={{ borderTop: "1px solid var(--border)", paddingTop: "16px" }}>
          <h4 style={{ marginBottom: "12px" }}>Custom Weight Ranking Results</h4>
          <div style={{ display: "flex", flexDirection: "column", gap: "8px" }}>
            {reRankedList.map((cand) => {
              // Find matching search candidate details
              const match = searchResponse.candidates.find(c => c.candidate_id === cand.candidate_id);
              const title = match?.current_title || "Unknown Title";
              const company = match?.current_company || "Unknown Company";

              // Delta computation
              const delta = cand.rank_delta;
              const hasDelta = delta !== 0;
              const isPositive = delta > 0;
              
              return (
                <div
                  key={cand.candidate_id}
                  style={{
                    display: "flex",
                    alignItems: "center",
                    padding: "8px 12px",
                    border: "1px solid var(--border)",
                    borderRadius: "var(--radius)",
                    backgroundColor: "var(--surface-1)",
                  }}
                >
                  <div style={{ flex: "0 0 50px", fontWeight: "bold" }}>
                    #{cand.new_rank}
                  </div>
                  
                  <div style={{ flex: "0 0 60px" }}>
                    {hasDelta ? (
                      <span style={{ color: isPositive ? "var(--text-success)" : "var(--text-danger)", fontWeight: 600 }}>
                        {isPositive ? `+${delta}` : `-${Math.abs(delta)}`}
                      </span>
                    ) : (
                      <span style={{ color: "var(--text-muted)" }}>—</span>
                    )}
                  </div>
                  
                  <div style={{ flex: 1, overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }}>
                    <span style={{ fontWeight: 600 }}>{title}</span>
                    <span style={{ color: "var(--text-secondary)", marginLeft: "8px" }}>at {company}</span>
                  </div>
                </div>
              );
            })}
          </div>
        </div>
      )}
      
      {loading && <div style={{ textAlign: "center", color: "var(--text-muted)" }}>Updating ranking...</div>}
    </div>
  );
}
