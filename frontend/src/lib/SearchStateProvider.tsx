"use client";

import React, { createContext, useContext, useState, ReactNode } from "react";
import type { SearchResponse, CandidateCardResponse } from "./types";

interface ScenarioInteraction {
  group: string;
  value: number;
  originalRank: number;
  newRank: number;
  candidateTitle: string;
}

interface SearchStateContextType {
  searchResponse: SearchResponse | null;
  setSearchResponse: (res: SearchResponse | null) => void;
  lastQuery: string;
  setLastQuery: (query: string) => void;
  shortlistedCandidates: CandidateCardResponse[];
  toggleShortlist: (candidate: CandidateCardResponse) => void;
  searchLatency: number | null;
  setSearchLatency: (latency: number | null) => void;
  scenarioInteraction: ScenarioInteraction | null;
  setScenarioInteraction: (interaction: ScenarioInteraction | null) => void;
}

const SearchStateContext = createContext<SearchStateContextType | undefined>(undefined);

export function SearchStateProvider({ children }: { children: ReactNode }) {
  const [searchResponse, setSearchResponse] = useState<SearchResponse | null>(null);
  const [lastQuery, setLastQuery] = useState<string>("");
  const [shortlistedCandidates, setShortlistedCandidates] = useState<CandidateCardResponse[]>([]);
  const [searchLatency, setSearchLatency] = useState<number | null>(null);
  const [scenarioInteraction, setScenarioInteraction] = useState<ScenarioInteraction | null>(null);

  const toggleShortlist = (candidate: CandidateCardResponse) => {
    setShortlistedCandidates((prev) => {
      const exists = prev.some((c) => c.candidate_id === candidate.candidate_id);
      if (exists) {
        return prev.filter((c) => c.candidate_id !== candidate.candidate_id);
      } else {
        return [...prev, candidate];
      }
    });
  };

  return (
    <SearchStateContext.Provider
      value={{
        searchResponse,
        setSearchResponse,
        lastQuery,
        setLastQuery,
        shortlistedCandidates,
        toggleShortlist,
        searchLatency,
        setSearchLatency,
        scenarioInteraction,
        setScenarioInteraction,
      }}
    >
      {children}
    </SearchStateContext.Provider>
  );
}

export function useSearchState() {
  const context = useContext(SearchStateContext);
  if (context === undefined) {
    throw new Error("useSearchState must be used within a SearchStateProvider");
  }
  return context;
}
