"use client";

import React, { useState } from "react";
import { useRouter } from "next/navigation";
import { Cpu, Database, Server, Brain, ListOrdered, ShieldCheck } from "lucide-react";
import { search } from "@/lib/api";
import { useSearchState } from "@/lib/SearchStateProvider";
import { CandidateCard } from "@/components/CandidateCard";

const SUGGESTIONS = [
  { icon: Cpu, label: "Senior ML Engineer", query: "Senior ML Engineer, 5+ years, PyTorch, NLP" },
  { icon: Database, label: "Data Scientist", query: "Data Scientist, 3+ years, experimentation" },
  { icon: Server, label: "Backend Engineer", query: "Backend Engineer, 4+ years, distributed systems" },
  { icon: Brain, label: "Staff AI Engineer", query: "Staff AI Engineer, 8+ years, LLM infra" },
  { icon: ListOrdered, label: "Top 10, ML background", query: "Top 10 candidates with ML background" },
  { icon: ShieldCheck, label: "Most trustworthy", query: "Most trustworthy senior profiles" },
];

export default function Home() {
  const [inputValue, setInputValue] = useState("");
  const [loading, setLoading] = useState(false);
  const [errorMsg, setErrorMsg] = useState<string | null>(null);
  
  const { searchResponse, setSearchResponse, setLastQuery, setSearchLatency, setScenarioInteraction } = useSearchState();
  const router = useRouter();

  const handleSearch = async (query: string) => {
    if (query.length < 20) {
      setErrorMsg("Job description is too short. Please provide at least 20 characters.");
      return;
    }
    
    setLoading(true);
    setErrorMsg(null);
    setInputValue(query);
    
    const startTime = performance.now();
    try {
      const res = await search({ jd_text: query });
      const endTime = performance.now();
      
      setSearchLatency(Math.round(endTime - startTime));
      setScenarioInteraction(null); // Reset on new search
      setSearchResponse(res);
      setLastQuery(query);
    } catch (e: unknown) {
      setErrorMsg((e as Error).message || "An unknown error occurred during search.");
    } finally {
      setLoading(false);
    }
  };

  const handleKeyDown = (e: React.KeyboardEvent<HTMLInputElement>) => {
    if (e.key === "Enter") {
      handleSearch(inputValue);
    }
  };

  return (
    <main style={{ maxWidth: "800px", margin: "0 auto", padding: "64px 16px" }}>
      <div style={{ textAlign: "center", marginBottom: "32px" }}>
        <h1 style={{ fontSize: "1.2rem", fontWeight: 600, color: "var(--text-muted)" }}>Candidate discovery</h1>
      </div>

      <div style={{ position: "relative", marginBottom: "12px" }}>
        <input
          type="text"
          value={inputValue}
          onChange={(e) => setInputValue(e.target.value)}
          onKeyDown={handleKeyDown}
          placeholder="Describe the role, or try a suggestion below"
          disabled={loading}
          style={{
            width: "100%",
            padding: "16px 24px",
            borderRadius: "999px",
            border: "1px solid var(--border)",
            fontSize: "1.1rem",
            boxShadow: "0 4px 12px rgba(0,0,0,0.05)",
            outline: "none",
            backgroundColor: "var(--surface-0)",
            color: "var(--text-primary)",
          }}
        />
        {loading && (
          <div style={{ position: "absolute", right: "24px", top: "50%", transform: "translateY(-50%)", color: "var(--text-muted)" }}>
            Searching...
          </div>
        )}
      </div>

      {errorMsg && (
        <div style={{ color: "var(--text-danger)", marginBottom: "24px", textAlign: "center" }}>
          {errorMsg}
        </div>
      )}

      {/* Preview Section */}
      {searchResponse && !loading && (
        <div style={{ marginBottom: "32px", padding: "24px", backgroundColor: "var(--surface-1)", borderRadius: "var(--radius)" }}>
          <div style={{ marginBottom: "16px", color: "var(--text-secondary)", fontWeight: 500 }}>
            Showing results for &quot;{inputValue}&quot;
          </div>
          <div style={{ display: "flex", flexDirection: "column", gap: "8px", marginBottom: "16px" }}>
            {searchResponse.candidates.slice(0, 3).map(cand => (
              <CandidateCard key={cand.candidate_id} candidate={cand} isInteractive={false} />
            ))}
          </div>
          <button
            onClick={() => router.push("/discovery")}
            style={{
              padding: "10px 16px",
              backgroundColor: "var(--bg-accent)",
              color: "var(--text-accent)",
              border: "none",
              borderRadius: "var(--radius)",
              fontWeight: 600,
              cursor: "pointer",
              width: "100%"
            }}
          >
            View all {searchResponse.candidates.length} results →
          </button>
        </div>
      )}

      <div style={{ display: "grid", gridTemplateColumns: "repeat(2, 1fr)", gap: "12px" }}>
        {SUGGESTIONS.map((s, idx) => (
          <button
            key={idx}
            onClick={() => handleSearch(s.query)}
            disabled={loading}
            style={{
              display: "flex",
              alignItems: "center",
              gap: "12px",
              padding: "16px",
              border: "1px solid var(--border)",
              borderRadius: "var(--radius)",
              backgroundColor: "var(--surface-0)",
              cursor: loading ? "default" : "pointer",
              textAlign: "left",
              color: "var(--text-secondary)",
              transition: "background-color 0.2s",
            }}
          >
            <s.icon size={20} color="var(--text-muted)" />
            <span style={{ fontWeight: 500 }}>{s.label}</span>
          </button>
        ))}
      </div>
    </main>
  );
}
