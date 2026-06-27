"use client";

import React, { createContext, useContext, useState, ReactNode } from "react";
import type { SearchResponse } from "./types";

interface SearchStateContextType {
  searchResponse: SearchResponse | null;
  setSearchResponse: (res: SearchResponse | null) => void;
  lastQuery: string;
  setLastQuery: (query: string) => void;
}

const SearchStateContext = createContext<SearchStateContextType | undefined>(undefined);

export function SearchStateProvider({ children }: { children: ReactNode }) {
  const [searchResponse, setSearchResponse] = useState<SearchResponse | null>(null);
  const [lastQuery, setLastQuery] = useState<string>("");

  return (
    <SearchStateContext.Provider value={{ searchResponse, setSearchResponse, lastQuery, setLastQuery }}>
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
