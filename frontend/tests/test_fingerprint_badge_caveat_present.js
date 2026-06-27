import React from "react";
import { render, screen, waitFor } from "@testing-library/react";
import CandidateDetail from "../src/app/candidate/[id]/page";
import { SearchStateProvider } from "../src/lib/SearchStateProvider";
import { FINGERPRINT_CAVEAT } from "../src/lib/constants";
import * as api from "../src/lib/api";

// Mock the router and standard navigation
jest.mock("next/navigation", () => ({
  useRouter: () => ({ push: jest.fn() })
}));

// Mock the API response
jest.mock("../src/lib/api", () => ({
  getCandidate: jest.fn()
}));

const mockCandidate = {
  candidate_id: "test_1",
  fingerprint_holder: true,
  profile: {
    current_title: "Staff AI Engineer",
    current_company: "Veritalent",
    years_of_experience: 8,
    location: "Remote"
  },
  shap_attribution: {
    top_features: []
  },
  trust_level: "high",
  trust_score: 9.5,
  trust_breakdown: {
    caveat: "Trust caveat.",
    checks: {}
  }
};

describe("Fingerprint Badge Compliance Test", () => {
  it("renders the verbatim FINGERPRINT_CAVEAT when fingerprint_holder is true", async () => {
    // Setup API mock to return a fingerprint holder
    (api.getCandidate as jest.Mock).mockResolvedValue(mockCandidate);

    // Render the Candidate Detail page wrapped in the Context provider
    // Simulate params resolution via Promise
    const paramsPromise = Promise.resolve({ id: "test_1" });
    
    render(
      <SearchStateProvider>
        <CandidateDetail params={paramsPromise} />
      </SearchStateProvider>
    );

    // Wait for data to load
    await waitFor(() => {
      expect(screen.getByText("Rare skill pattern")).toBeInTheDocument();
    });

    // The core compliance assertion: The EXACT FINGERPRINT_CAVEAT string MUST be present in the DOM
    expect(screen.getByText(FINGERPRINT_CAVEAT, { exact: false })).toBeInTheDocument();
  });

  it("does not render the badge or caveat when fingerprint_holder is false", async () => {
    // Setup API mock to return a NON-fingerprint holder
    (api.getCandidate as jest.Mock).mockResolvedValue({
      ...mockCandidate,
      fingerprint_holder: false,
    });

    const paramsPromise = Promise.resolve({ id: "test_2" });
    
    render(
      <SearchStateProvider>
        <CandidateDetail params={paramsPromise} />
      </SearchStateProvider>
    );

    await waitFor(() => {
      // Ensure the title loaded so we know render is complete
      expect(screen.getByText("Staff AI Engineer")).toBeInTheDocument();
    });

    // Assert the badge and caveat are completely absent
    expect(screen.queryByText("Rare skill pattern")).not.toBeInTheDocument();
    expect(screen.queryByText(FINGERPRINT_CAVEAT, { exact: false })).not.toBeInTheDocument();
  });
});
