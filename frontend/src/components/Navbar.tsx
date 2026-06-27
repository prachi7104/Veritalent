"use client";

import React from "react";
import Link from "next/link";
import { usePathname } from "next/navigation";
import { useSearchState } from "@/lib/SearchStateProvider";
import { Star, Home, Search, Play } from "lucide-react";

export function Navbar() {
  const pathname = usePathname();
  const { shortlistedCandidates, searchResponse } = useSearchState();

  const navItems = [
    { href: "/", label: "Search", icon: Home },
    ...(searchResponse ? [{ href: "/discovery", label: "Discovery", icon: Search }] : []),
    { href: "/shortlist", label: `Shortlist (${shortlistedCandidates.length})`, icon: Star },
    { href: "/demo", label: "Demo Mode", icon: Play },
  ];

  return (
    <nav
      style={{
        display: "flex",
        alignItems: "center",
        justifyContent: "space-between",
        padding: "16px 24px",
        backgroundColor: "var(--surface-0)",
        borderBottom: "1px solid var(--border)",
        maxWidth: "1200px",
        margin: "0 auto",
        width: "100%",
      }}
    >
      <div style={{ fontWeight: 700, fontSize: "1.1rem", color: "var(--text-primary)" }}>
        Veritalent
      </div>
      <div style={{ display: "flex", gap: "24px" }}>
        {navItems.map((item) => {
          const isActive = pathname === item.href;
          const Icon = item.icon;
          return (
            <Link
              key={item.href}
              href={item.href}
              style={{
                display: "flex",
                alignItems: "center",
                gap: "6px",
                textDecoration: "none",
                fontSize: "0.95rem",
                fontWeight: isActive ? 600 : 500,
                color: isActive ? "var(--text-accent)" : "var(--text-secondary)",
                transition: "color 0.2s",
              }}
            >
              <Icon size={16} />
              <span>{item.label}</span>
            </Link>
          );
        })}
      </div>
    </nav>
  );
}
