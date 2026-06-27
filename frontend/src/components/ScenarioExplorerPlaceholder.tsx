import React from "react";
import { Settings2 } from "lucide-react";

export function ScenarioExplorerPlaceholder() {
  return (
    <div style={{ marginTop: "32px", borderTop: "1px solid var(--border)", paddingTop: "16px" }}>
      <button 
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
  );
}
