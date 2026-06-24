from typing import List, Dict, Any
import networkx as nx
import pickle
import os

class SkillGraph:
    def __init__(self):
        self.graph = nx.Graph()

    def build(self, candidates: List[Dict[str, Any]]):
        print("Building skill adjacency graph...")
        co_occurrences = {}
        
        for cand in candidates:
            skills = []
            for skill in cand.get("skills", []):
                name = skill.get("name")
                if name:
                    skills.append(name)
            
            # Count co-occurrences
            for i in range(len(skills)):
                for j in range(i + 1, len(skills)):
                    s1, s2 = skills[i], skills[j]
                    # order them to avoid double counting
                    if s1 > s2:
                        s1, s2 = s2, s1
                    pair = (s1, s2)
                    co_occurrences[pair] = co_occurrences.get(pair, 0) + 1
                    
        # Add to graph
        for (s1, s2), count in co_occurrences.items():
            self.graph.add_edge(s1, s2, weight=count)
            
        print(f"Graph built with {self.graph.number_of_nodes()} nodes and {self.graph.number_of_edges()} edges.")

    def get_nearest_skills(self, target_skill: str, top_k: int = 5) -> Dict[str, float]:
        """
        Returns a dict of {skill_name: affinity_score} for the target skill.
        Affinity score is based on the co-occurrence edge weight.
        """
        if target_skill not in self.graph:
            return {}
            
        neighbors = self.graph[target_skill]
        # Sort by weight descending
        sorted_neighbors = sorted(neighbors.items(), key=lambda x: x[1]['weight'], reverse=True)
        
        results = {}
        # Normalize weights against max weight for this skill
        max_weight = sorted_neighbors[0][1]['weight'] if sorted_neighbors else 1.0
        
        for neighbor, data in sorted_neighbors[:top_k]:
            results[neighbor] = data['weight'] / max_weight
            
        return results

    def score_candidate(self, candidate: Dict[str, Any], required_skills: List[str]) -> float:
        """
        Scores a candidate based on skill graph transferability.
        If they have the required skill, score += 1.
        If not, check their skills against nearest neighbors of the required skill.
        """
        cand_skills = {s.get("name") for s in candidate.get("skills", []) if "name" in s}
        score = 0.0
        
        for req in required_skills:
            if req in cand_skills:
                score += 1.0
            else:
                nearest = self.get_nearest_skills(req)
                best_partial = 0.0
                for cand_skill in cand_skills:
                    if cand_skill in nearest:
                        if nearest[cand_skill] > best_partial:
                            best_partial = nearest[cand_skill]
                score += best_partial * 0.5 # Partial credit multiplier
                
        return score

    def save(self, path: str):
        print(f"Saving SkillGraph to {path}...")
        with open(path, 'wb') as f:
            pickle.dump(self.graph, f)
            
    def load(self, path: str):
        print(f"Loading SkillGraph from {path}...")
        with open(path, 'rb') as f:
            self.graph = pickle.load(f)
