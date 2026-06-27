import React from "react";
import type { TrustLevel } from "@/lib/types";
import { TRUST_LEVEL_COLOR } from "@/lib/constants";

interface TrustBadgeProps {
  level: TrustLevel;
  score?: number;
}

export function TrustBadge({ level, score }: TrustBadgeProps) {
  const color = TRUST_LEVEL_COLOR[level];
  const label = level.charAt(0).toUpperCase() + level.slice(1);

  return (
    <div
      style={{
        backgroundColor: color.bg,
        color: color.text,
        padding: "4px 8px",
        borderRadius: "var(--radius)",
        fontSize: "0.85rem",
        fontWeight: 600,
        display: "inline-flex",
        alignItems: "center",
        gap: "4px",
      }}
    >
      <span>{label}</span>
      {score !== undefined && (
        <span style={{ opacity: 0.8 }}>· {score.toFixed(2)}</span>
      )}
    </div>
  );
}
