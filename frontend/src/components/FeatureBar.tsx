import React from "react";
import type { FeatureContribution } from "@/lib/types";

interface FeatureBarProps {
  feature: FeatureContribution;
  maxAbsValue: number;
}

export function FeatureBar({ feature, maxAbsValue }: FeatureBarProps) {
  // Positive extends right, negative extends left
  // Max width is 50% on either side to fit in container
  const normalizedWidth = (Math.abs(feature.shap_contribution) / maxAbsValue) * 50;
  
  const isPositive = feature.direction === "positive";
  
  return (
    <div style={{ display: "flex", alignItems: "center", marginBottom: "8px", fontSize: "0.9rem" }}>
      <div style={{ width: "25%", fontFamily: "var(--font-mono)", textAlign: "right", paddingRight: "16px", color: "var(--text-secondary)" }}>
        {feature.feature_name}
      </div>
      <div style={{ flex: 1, display: "flex", alignItems: "center", position: "relative", height: "24px" }}>
        {/* Center axis */}
        <div style={{ position: "absolute", left: "50%", top: 0, bottom: 0, width: "1px", backgroundColor: "var(--border-strong)" }} />
        
        {/* Bar */}
        <div
          style={{
            position: "absolute",
            height: "16px",
            backgroundColor: isPositive ? "#3b82f6" : "#ef4444", // Using raw hex for chart fills as per spec
            width: `${normalizedWidth}%`,
            ...(isPositive ? { left: "50%", borderRadius: "0 4px 4px 0" } : { right: "50%", borderRadius: "4px 0 0 4px" })
          }}
        />
      </div>
      <div style={{ width: "15%", textAlign: "right", paddingLeft: "16px", fontFamily: "var(--font-mono)", color: "var(--text-muted)" }}>
        {feature.value.toFixed(2)}
      </div>
    </div>
  );
}
