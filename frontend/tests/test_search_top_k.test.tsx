import React from "react";
import { render, screen, fireEvent, waitFor } from "@testing-library/react";
import Home from "../src/app/page";
import Discovery from "../src/app/discovery/page";
import { SearchStateProvider, useSearchState } from "../src/lib/SearchStateProvider";
import * as api from "../src/lib/api";

// Mock the router
jest.mock("next/navigation", () => ({
  useRouter: () => ({ push: jest.fn() })
}));

// Mock the API responses
jest.mock("../src/lib/api", () => ({
  search: jest.fn(),
  rerankByJD: jest.fn(),
  getScenarios: jest.fn().mockResolvedValue([])
}));

const mockCandidates = Array.from({ length: 100 }, (_, i) => ({
  candidate_id: `cand_${i}`,
  rank: i + 1,
  score: 1.0 - i * 0.01,
  current_title: "Staff AI Engineer",
  current_company: "Veritalent",
  years_of_experience: 8,
  location: "Remote",
  top_features: [],
  matched_deep_ir_skills: [],
  missing_deep_ir_skills: [],
  trust_score: 9.0,
  trust_level: "high"
}));

const mockSearchResponse = {
  session_id: "test_session_id",
  candidates: mockCandidates,
  funnel_stats: {
    total_pool: 100000,
    title_relevant: 30000,
    retrieved: 200,
    shown: 100
  },
  jd_decomposition: {
    must_haves: ["React"],
    nice_to_haves: [],
    hard_exclusions: [],
    experience_band: "Unknown",
    logistics: {}
  }
};

const DiscoveryWithState = ({ initialResponse }: { initialResponse: typeof mockSearchResponse }) => {
  const { setSearchResponse, setLastQuery } = useSearchState();
  React.useEffect(() => {
    setSearchResponse(initialResponse);
    setLastQuery("Expert python engineer for ranking systems. ");
  }, [initialResponse, setSearchResponse, setLastQuery]);
  return <Discovery />;
};

describe("Frontend search and rerank top_k validation", () => {
  beforeEach(() => {
    jest.clearAllMocks();
    jest.useFakeTimers();
  });

  afterEach(() => {
    jest.useRealTimers();
  });

  it("calls search API with top_k = 100 when search is triggered on Home page", async () => {
    (api.search as jest.Mock).mockResolvedValue(mockSearchResponse);

    render(
      <SearchStateProvider>
        <Home />
      </SearchStateProvider>
    );

    const input = screen.getByPlaceholderText(/Describe the role/i);
    fireEvent.change(input, { target: { value: "We need a senior frontend engineer with 5 years experience" } });
    fireEvent.keyDown(input, { key: "Enter", code: "Enter", charCode: 13 });

    await waitFor(() => {
      expect(api.search).toHaveBeenCalledWith({
        jd_text: "We need a senior frontend engineer with 5 years experience",
        top_k: 100
      });
    });
  });

  it("calls rerankByJD API with top_k matching candidates length on Discovery page", async () => {
    (api.rerankByJD as jest.Mock).mockResolvedValue(mockSearchResponse);

    render(
      <SearchStateProvider>
        <DiscoveryWithState initialResponse={mockSearchResponse} />
      </SearchStateProvider>
    );

    // Wait for the discovery component to render
    await screen.findByText(/100,000 total/);

    // Click edit button
    const editButton = screen.getByRole("button", { name: "Edit" });
    fireEvent.click(editButton);

    // Get textarea and type new JD (min length 20)
    const textarea = screen.getByRole("textbox");
    fireEvent.change(textarea, { target: { value: "Updated job description text with more than twenty characters." } });

    // Advance fake timers by 600ms debounce
    jest.advanceTimersByTime(600);

    await waitFor(() => {
      expect(api.rerankByJD).toHaveBeenCalledWith({
        session_id: "test_session_id",
        updated_jd_text: "Updated job description text with more than twenty characters.",
        top_k: 100
      });
    });
  });
});
