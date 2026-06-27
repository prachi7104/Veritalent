"use client";

import React from "react";
import { useRouter } from "next/navigation";
import { useSearchState } from "@/lib/SearchStateProvider";
import { TrustBadge } from "@/components/TrustBadge";
import { Clock, Sliders, Award, BarChart2 } from "lucide-react";

export default function DemoMode() {
  const router = useRouter();
  const { searchResponse, searchLatency, scenarioInteraction } = useSearchState();

  // 1. Cold-State Gate
  if (!searchResponse) {
    return (
      <div style={{ maxWidth: "600px", margin: "64px auto", textAlign: "center" }}>
        <h2 style={{ marginBottom: "16px", color: "var(--text-primary)" }}>Demo Mode is inactive</h2>
        <p style={{ color: "var(--text-secondary)", marginBottom: "24px" }}>
          Run a search to populate the Demo Mode dashboard.
        </p>
        <button
          onClick={() => router.push("/")}
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
          Go to Search
        </button>
      </div>
    );
  }

  const { funnel_stats, candidates } = searchResponse;
  
  // Spotlight candidate (Rank #1)
  const spotlightCand = candidates.find(c => c.rank === 1);

  // Spotlight highest-magnitude feature attribution
  let highestFeatureText = "";
  if (spotlightCand && spotlightCand.top_features && spotlightCand.top_features.length > 0) {
    let bestFeat = spotlightCand.top_features[0];
    spotlightCand.top_features.forEach((f) => {
      if (Math.abs(f.shap_contribution) > Math.abs(bestFeat.shap_contribution)) {
        bestFeat = f;
      }
    });

    const isPositive = bestFeat.shap_contribution >= 0;
    highestFeatureText = `top reason: ${bestFeat.feature_name} ${isPositive ? "+" : ""}${bestFeat.shap_contribution.toFixed(2)}`;
  }

  // Latency Target Computations
  const latency = searchLatency || 0;
  let latencyColor = "var(--text-danger)";
  let latencyBg = "var(--bg-danger)";
  let latencyLabel = "Slow latency bounds exceeded";
  
  if (latency > 0) {
    if (latency < 300) {
      latencyColor = "var(--text-success)";
      latencyBg = "var(--bg-success)";
      latencyLabel = "within the 300ms demo target";
    } else if (latency <= 800) {
      latencyColor = "var(--text-warning)";
      latencyBg = "var(--bg-warning)";
      latencyLabel = "above the 300ms demo target, under the 800ms ceiling";
    }
  }

  // Funnel calculations
  const total = funnel_stats.total_pool || 100000;
  const relevant = funnel_stats.title_relevant || 31179;
  const shown = funnel_stats.shown || 20;

  const relevantWidth = (relevant / total) * 100;
  const shownWidth = Math.max((shown / total) * 100, 1.5); // Ensure thin line is visible

  return (
    <div style={{ maxWidth: "800px", margin: "0 auto", padding: "32px 16px" }}>
      <h1 style={{ fontSize: "1.50rem", marginBottom: "32px", display: "flex", alignItems: "center", gap: "8px" }}>
        <span>Judges Pitch Dashboard</span>
      </h1>

      {/* 1. Latency Badge */}
      <div style={{ marginBottom: "24px", padding: "16px", backgroundColor: "var(--surface-0)", borderRadius: "var(--radius)", border: "1px solid var(--border)", display: "flex", alignItems: "center", gap: "12px" }}>
        <Clock size={20} color="var(--text-secondary)" />
        <div style={{ display: "flex", alignItems: "center", gap: "8px", flexWrap: "wrap" }}>
          <span style={{ fontWeight: 600 }}>Latency Check:</span>
          {latency > 0 ? (
            <span style={{ padding: "4px 10px", backgroundColor: latencyBg, color: latencyColor, borderRadius: "var(--radius)", fontSize: "0.85rem", fontWeight: 700 }}>
              {latency}ms
            </span>
          ) : (
            <span style={{ color: "var(--text-muted)", fontSize: "0.9rem" }}>No active latency data</span>
          )}
          {latency > 0 && <span style={{ fontSize: "0.9rem", color: "var(--text-secondary)" }}>{latencyLabel}</span>}
        </div>
      </div>

      {/* 2. Compact Funnel */}
      <div style={{ marginBottom: "24px", padding: "24px", backgroundColor: "var(--surface-0)", borderRadius: "var(--radius)", border: "1px solid var(--border)" }}>
        <div style={{ display: "flex", alignItems: "center", gap: "8px", marginBottom: "16px" }}>
          <BarChart2 size={20} color="var(--text-secondary)" />
          <h3 style={{ margin: 0, fontSize: "1.05rem" }}>Funnel Analysis</h3>
        </div>
        
        <div style={{ display: "flex", flexDirection: "column", gap: "12px" }}>
          {/* Bar 1: Total */}
          <div>
            <div style={{ height: "6px", backgroundColor: "#93c5fd", width: "100%", borderRadius: "999px" }} />
            <div style={{ display: "flex", justifyContent: "space-between", fontSize: "0.85rem", marginTop: "4px", color: "var(--text-secondary)" }}>
              <span>Total Candidate Pool</span>
              <span style={{ fontFamily: "var(--font-mono)", fontWeight: 600 }}>{total.toLocaleString()}</span>
            </div>
          </div>
          
          {/* Bar 2: Relevant */}
          <div>
            <div style={{ height: "6px", backgroundColor: "#60a5fa", width: `${relevantWidth}%`, borderRadius: "999px" }} />
            <div style={{ display: "flex", justifyContent: "space-between", fontSize: "0.85rem", marginTop: "4px", color: "var(--text-secondary)" }}>
              <span>Title Relevant (Keyword Match)</span>
              <span style={{ fontFamily: "var(--font-mono)", fontWeight: 600 }}>{relevant.toLocaleString()}</span>
            </div>
          </div>
          
          {/* Bar 3: Shown */}
          <div>
            <div style={{ height: "6px", backgroundColor: "#2563eb", width: `${shownWidth}%`, borderRadius: "999px" }} />
            <div style={{ display: "flex", justifyContent: "space-between", fontSize: "0.85rem", marginTop: "4px", color: "var(--text-secondary)" }}>
              <span>Retrieved & Shown</span>
              <span style={{ fontFamily: "var(--font-mono)", fontWeight: 600 }}>{shown.toLocaleString()}</span>
            </div>
          </div>
        </div>
      </div>

      {/* 3. Spotlight Block */}
      {spotlightCand && (
        <div style={{ marginBottom: "24px", padding: "24px", backgroundColor: "var(--surface-0)", borderRadius: "var(--radius)", border: "1px solid var(--border)" }}>
          <div style={{ display: "flex", alignItems: "center", gap: "8px", marginBottom: "16px" }}>
            <Award size={20} color="var(--text-secondary)" />
            <h3 style={{ margin: 0, fontSize: "1.05rem" }}>Rank #1 Spotlight</h3>
          </div>
          
          <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", flexWrap: "wrap", gap: "12px", borderBottom: highestFeatureText ? "1px solid var(--border)" : "none", paddingBottom: highestFeatureText ? "16px" : "0", marginBottom: highestFeatureText ? "16px" : "0" }}>
            <div>
              <div style={{ fontSize: "1.2rem", fontWeight: 700, color: "var(--text-primary)" }}>
                {spotlightCand.current_title}
              </div>
              <div style={{ fontSize: "0.95rem", color: "var(--text-secondary)", marginTop: "4px" }}>
                {spotlightCand.current_company} · {spotlightCand.years_of_experience} yrs · {spotlightCand.location}
              </div>
            </div>
            <TrustBadge level={spotlightCand.trust_level} />
          </div>
          
          {highestFeatureText && (
            <div style={{ fontSize: "0.95rem", fontFamily: "var(--font-mono)", color: "var(--text-accent)", fontWeight: 600 }}>
              {highestFeatureText}
            </div>
          )}
        </div>
      )}

      {/* 4. Scenario Example Block */}
      <div style={{ padding: "24px", backgroundColor: "var(--surface-0)", borderRadius: "var(--radius)", border: "1px solid var(--border)" }}>
        <div style={{ display: "flex", alignItems: "center", gap: "8px", marginBottom: "16px" }}>
          <Sliders size={20} color="var(--text-secondary)" />
          <h3 style={{ margin: 0, fontSize: "1.05rem" }}>Scenario Validation</h3>
        </div>
        
        {scenarioInteraction ? (
          <div style={{ fontSize: "1rem", lineHeight: 1.6, color: "var(--text-primary)" }}>
            Raising the <span style={{ fontWeight: 600, color: "var(--text-accent)" }}>{scenarioInteraction.group}</span> weight to{" "}
            <span style={{ fontWeight: 600 }}>{scenarioInteraction.value}</span> moved{" "}
            <span style={{ fontWeight: 600 }}>{scenarioInteraction.candidateTitle}</span> from{" "}
            <span style={{ textDecoration: "line-through", color: "var(--text-muted)", marginRight: "4px" }}>#{scenarioInteraction.originalRank}</span>
            to <span style={{ fontWeight: 700, color: "var(--text-success)" }}>#{scenarioInteraction.newRank}</span>.
          </div>
        ) : (
          <div style={{ fontSize: "0.95rem", color: "var(--text-muted)", fontStyle: "italic" }}>
            Try the scenario explorer to see this update live.
          </div>
        )}
      </div>
    </div>
  );
}
