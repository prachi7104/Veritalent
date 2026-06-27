import React from "react";
import { useRouter } from "next/navigation";
import type { CandidateCardResponse } from "@/lib/types";
import { TrustBadge } from "./TrustBadge";
import { TIED_BAND_RANKS } from "@/lib/constants";

interface CandidateCardProps {
  candidate: CandidateCardResponse;
  isInteractive?: boolean;
  showCheckbox?: boolean;
  checked?: boolean;
  onCheckChange?: (checked: boolean) => void;
  disabledCheckbox?: boolean;
}

export function CandidateCard({
  candidate,
  isInteractive = true,
  showCheckbox = false,
  checked = false,
  onCheckChange,
  disabledCheckbox = false,
}: CandidateCardProps) {
  const router = useRouter();

  const handleClick = (e: React.MouseEvent) => {
    // Prevent triggering navigation when clicking the checkbox
    if ((e.target as HTMLElement).tagName === "INPUT") {
      return;
    }
    if (isInteractive) {
      router.push(`/candidate/${encodeURIComponent(candidate.candidate_id)}`);
    }
  };

  const isTied = TIED_BAND_RANKS.has(candidate.rank);
  const gapCount = candidate.skill_gap?.missing_deep_ir_skills?.length || 0;

  return (
    <div
      onClick={handleClick}
      style={{
        display: "flex",
        alignItems: "center",
        padding: "16px",
        border: "1px solid var(--border)",
        borderRadius: "var(--radius)",
        backgroundColor: "var(--surface-0)",
        cursor: isInteractive ? "pointer" : "default",
        marginBottom: "12px",
        transition: "border-color 0.2s",
        ...(isInteractive && { ":hover": { borderColor: "var(--border-strong)" } }),
      }}
    >
      {showCheckbox && (
        <div style={{ marginRight: "16px", display: "flex", alignItems: "center" }}>
          <input
            type="checkbox"
            checked={checked}
            disabled={disabledCheckbox && !checked}
            onChange={(e) => onCheckChange?.(e.target.checked)}
            style={{ width: "20px", height: "20px", cursor: (disabledCheckbox && !checked) ? "default" : "pointer" }}
          />
        </div>
      )}

      <div style={{ flex: "0 0 60px", fontSize: "1.5rem", fontWeight: "bold", color: "var(--text-muted)" }}>
        #{candidate.rank}
      </div>
      
      <div style={{ flex: 1, paddingLeft: "16px" }}>
        <div style={{ fontWeight: 600, fontSize: "1.1rem", color: "var(--text-primary)", marginBottom: "4px" }}>
          {candidate.current_title}
        </div>
        <div style={{ color: "var(--text-secondary)", fontSize: "0.95rem" }}>
          {candidate.current_company} · {candidate.years_of_experience} yrs · {candidate.location}
        </div>
        
        {(gapCount > 0 || isTied) && (
          <div style={{ display: "flex", gap: "12px", marginTop: "8px", fontSize: "0.85rem", color: "var(--text-muted)" }}>
            {gapCount > 0 && <span>{gapCount} skill gap{gapCount !== 1 ? 's' : ''}</span>}
            {isTied && <span>approximately tied within this band</span>}
          </div>
        )}
      </div>

      <div style={{ paddingLeft: "16px" }}>
        <TrustBadge level={candidate.trust_level} />
      </div>
    </div>
  );
}
