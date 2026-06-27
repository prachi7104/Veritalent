import type {
  SearchRequest, SearchResponse, RerankRequest, ScenarioRerankRequest,
  ScenarioRerankResponse, CandidateDetailResponse, CompareRequest,
  CompareResponse, HealthResponse,
} from "./types";

const BASE_URL = process.env.NEXT_PUBLIC_API_BASE_URL ?? "http://localhost:8000";

export class ApiError extends Error {
  status: number;
  constructor(status: number, message: string) {
    super(message);
    this.status = status;
  }
}

async function fetchWithHandleError(url: string, options?: RequestInit) {
  const response = await fetch(url, options);
  if (!response.ok) {
    let detailMessage = "An unknown error occurred";
    try {
      const errorData = await response.json();
      if (errorData && errorData.detail) {
        detailMessage = typeof errorData.detail === "string" 
            ? errorData.detail 
            : JSON.stringify(errorData.detail);
      }
    } catch {
      // Fallback if not JSON
      detailMessage = await response.text();
    }
    throw new ApiError(response.status, detailMessage);
  }
  return response.json();
}

export async function search(req: SearchRequest): Promise<SearchResponse> {
  return fetchWithHandleError(`${BASE_URL}/search`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(req),
  });
}

export async function rerankByJD(req: RerankRequest): Promise<SearchResponse> {
  return fetchWithHandleError(`${BASE_URL}/rerank`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(req),
  });
}

export async function scenarioRerank(req: ScenarioRerankRequest): Promise<ScenarioRerankResponse> {
  return fetchWithHandleError(`${BASE_URL}/scenarios/rerank`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(req),
  });
}

export async function getCandidate(id: string): Promise<CandidateDetailResponse> {
  return fetchWithHandleError(`${BASE_URL}/candidates/${encodeURIComponent(id)}`);
}

export async function compareCandidates(req: CompareRequest): Promise<CompareResponse> {
  return fetchWithHandleError(`${BASE_URL}/compare`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(req),
  });
}

export async function getHealth(): Promise<HealthResponse> {
  return fetchWithHandleError(`${BASE_URL}/health`);
}
