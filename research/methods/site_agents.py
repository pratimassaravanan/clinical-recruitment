"""Multi-Agent Site Negotiation Protocol.

Sites as independent agents with private utility functions and strategic negotiation.

Key features:
1. Site agents with private utilities (capacity, conversion rates, costs)
2. Negotiation protocol with offers, counteroffers, acceptance/rejection
3. Information asymmetry - sites know their true capacity, agent estimates
4. Strategic behavior - sites may bluff or delay based on competition
"""

from __future__ import annotations

import copy
import random
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple


class NegotiationOutcome(Enum):
    """Possible outcomes of a negotiation round."""
    PENDING = "pending"
    ACCEPTED = "accepted"
    REJECTED = "rejected"
    COUNTEROFFER = "counteroffer"
    EXPIRED = "expired"


@dataclass
class NegotiationOffer:
    """An offer in the negotiation protocol."""
    
    offer_id: str
    from_agent: str  # "recruiter" or site_id
    to_agent: str
    offered_capacity: int  # Capacity slots offered
    offered_payment: float  # Payment per enrollment
    valid_until_step: int  # Offer expires after this step
    conditions: Dict[str, Any] = field(default_factory=dict)  # Extra conditions


@dataclass
class NegotiationHistory:
    """History of negotiation for a site."""
    
    site_id: str
    offers: List[NegotiationOffer] = field(default_factory=list)
    outcomes: List[Tuple[str, NegotiationOutcome]] = field(default_factory=list)
    total_accepted_capacity: int = 0
    total_payment_committed: float = 0.0
    relationship_score: float = 0.5  # 0-1, affects future negotiations


@dataclass
class SiteAgentState:
    """Internal state of a site agent."""
    
    site_id: str
    # True private values (hidden from recruiter)
    true_capacity: int
    true_min_payment: float  # Minimum acceptable payment per enrollment
    true_conversion_rate: float
    true_retention_rate: float
    
    # Public/observable values
    reported_capacity: int  # May differ from true
    current_enrollment: int = 0
    
    # Strategic parameters
    risk_aversion: float = 0.5  # 0-1, higher = more conservative
    patience: float = 0.5  # 0-1, higher = more willing to wait
    competition_awareness: float = 0.5  # 0-1, higher = more strategic
    
    # Negotiation state
    pending_offers: List[NegotiationOffer] = field(default_factory=list)
    accepted_capacity: int = 0
    
    def available_capacity(self) -> int:
        """Actual available capacity."""
        return max(0, self.true_capacity - self.current_enrollment - self.accepted_capacity)
    
    def utility(self, payment: float, capacity_used: int) -> float:
        """Site's utility from accepting an offer."""
        if payment < self.true_min_payment:
            return -1.0  # Unacceptable
        
        profit_per = payment - self.true_min_payment
        capacity_cost = capacity_used / max(1, self.true_capacity) * self.risk_aversion
        
        return profit_per * capacity_used - capacity_cost * 100


class SiteAgent:
    """An independent site agent with private utilities and strategic behavior."""
    
    def __init__(
        self,
        site_id: str,
        true_capacity: int = 50,
        true_min_payment: float = 500.0,
        true_conversion_rate: float = 0.7,
        true_retention_rate: float = 0.85,
        risk_aversion: float = 0.5,
        patience: float = 0.5,
        competition_awareness: float = 0.5,
        seed: Optional[int] = None,
    ):
        self.state = SiteAgentState(
            site_id=site_id,
            true_capacity=true_capacity,
            true_min_payment=true_min_payment,
            true_conversion_rate=true_conversion_rate,
            true_retention_rate=true_retention_rate,
            reported_capacity=int(true_capacity * random.uniform(0.8, 1.0)),  # May underreport
            risk_aversion=risk_aversion,
            patience=patience,
            competition_awareness=competition_awareness,
        )
        self.history = NegotiationHistory(site_id=site_id)
        self.rng = random.Random(seed)
    
    def receive_offer(
        self,
        offer: NegotiationOffer,
        current_step: int,
        market_context: Dict[str, Any],
    ) -> Tuple[NegotiationOutcome, Optional[NegotiationOffer]]:
        """Process an incoming offer and decide response."""
        
        # Check if offer expired
        if current_step > offer.valid_until_step:
            return NegotiationOutcome.EXPIRED, None
        
        # Check capacity
        if offer.offered_capacity > self.state.available_capacity():
            # Counter with available capacity
            if self.state.available_capacity() > 0:
                counter = self._make_counteroffer(offer, current_step)
                return NegotiationOutcome.COUNTEROFFER, counter
            return NegotiationOutcome.REJECTED, None
        
        # Evaluate utility
        utility = self.state.utility(offer.offered_payment, offer.offered_capacity)
        
        if utility < 0:
            # Payment too low - counter with higher price
            counter = self._make_counteroffer(offer, current_step)
            return NegotiationOutcome.COUNTEROFFER, counter
        
        # Strategic decision based on market context
        competition_factor = market_context.get("num_competing_sites", 3) / 3.0
        urgency_factor = market_context.get("enrollment_urgency", 0.5)
        
        # More patient sites wait for better offers if competition is low
        wait_threshold = self.state.patience * (1 - urgency_factor) * (1 - competition_factor)
        
        if self.rng.random() < wait_threshold and utility < 100:
            # Wait for better offer
            return NegotiationOutcome.PENDING, None
        
        # Accept the offer
        self.state.accepted_capacity += offer.offered_capacity
        self.state.pending_offers.append(offer)
        self.history.total_accepted_capacity += offer.offered_capacity
        self.history.total_payment_committed += offer.offered_payment * offer.offered_capacity
        self.history.outcomes.append((offer.offer_id, NegotiationOutcome.ACCEPTED))
        self.history.relationship_score = min(1.0, self.history.relationship_score + 0.05)
        
        return NegotiationOutcome.ACCEPTED, None
    
    def _make_counteroffer(
        self,
        original: NegotiationOffer,
        current_step: int,
    ) -> NegotiationOffer:
        """Generate a counteroffer."""
        
        # Determine counter capacity
        counter_capacity = min(
            original.offered_capacity,
            self.state.available_capacity()
        )
        
        # Determine counter payment (at least min payment + margin)
        margin = self.state.true_min_payment * (0.1 + 0.2 * self.state.risk_aversion)
        counter_payment = max(
            original.offered_payment * 1.1,
            self.state.true_min_payment + margin
        )
        
        return NegotiationOffer(
            offer_id=f"counter_{original.offer_id}_{current_step}",
            from_agent=self.state.site_id,
            to_agent=original.from_agent,
            offered_capacity=counter_capacity,
            offered_payment=counter_payment,
            valid_until_step=current_step + 5,
            conditions={"is_counteroffer": True, "original_offer": original.offer_id},
        )
    
    def update_enrollment(self, new_enrollments: int) -> None:
        """Update site's enrollment count."""
        self.state.current_enrollment += new_enrollments
    
    def get_public_info(self) -> Dict[str, Any]:
        """Information visible to the recruiter."""
        return {
            "site_id": self.state.site_id,
            "reported_capacity": self.state.reported_capacity,
            "current_enrollment": self.state.current_enrollment,
            "available_capacity_estimate": max(0, self.state.reported_capacity - self.state.current_enrollment),
            "conversion_rate_estimate": self.state.true_conversion_rate * random.uniform(0.9, 1.1),
            "retention_rate_estimate": self.state.true_retention_rate * random.uniform(0.9, 1.1),
            "relationship_score": self.history.relationship_score,
        }


class MultiAgentNegotiator:
    """Orchestrates negotiation between recruiter and multiple site agents."""
    
    def __init__(
        self,
        sites: Optional[Dict[str, Dict[str, Any]]] = None,
        seed: Optional[int] = None,
    ):
        self.rng = random.Random(seed)
        self.site_agents: Dict[str, SiteAgent] = {}
        self.offer_counter = 0
        self.step = 0
        
        # Initialize site agents from site config
        if sites:
            for site_id, site_config in sites.items():
                self.add_site_agent(site_id, site_config)
    
    def add_site_agent(
        self,
        site_id: str,
        config: Dict[str, Any],
    ) -> None:
        """Add a site agent from configuration."""
        self.site_agents[site_id] = SiteAgent(
            site_id=site_id,
            true_capacity=int(config.get("capacity_remaining", 50) * 1.2),
            true_min_payment=config.get("enrollment_cost", 500) * 0.8,
            true_conversion_rate=config.get("conversion_rate", 0.7),
            true_retention_rate=config.get("retention_rate", 0.85),
            risk_aversion=self.rng.uniform(0.3, 0.7),
            patience=self.rng.uniform(0.3, 0.7),
            competition_awareness=self.rng.uniform(0.3, 0.7),
            seed=self.rng.randint(0, 10000),
        )
    
    def make_offer(
        self,
        to_site: str,
        capacity_requested: int,
        payment_offered: float,
        valid_for_steps: int = 10,
    ) -> NegotiationOffer:
        """Create an offer to a site."""
        self.offer_counter += 1
        return NegotiationOffer(
            offer_id=f"offer_{self.offer_counter}",
            from_agent="recruiter",
            to_agent=to_site,
            offered_capacity=capacity_requested,
            offered_payment=payment_offered,
            valid_until_step=self.step + valid_for_steps,
        )
    
    def submit_offer(
        self,
        offer: NegotiationOffer,
        market_context: Optional[Dict[str, Any]] = None,
    ) -> Tuple[NegotiationOutcome, Optional[NegotiationOffer]]:
        """Submit an offer to a site and get response."""
        
        if offer.to_agent not in self.site_agents:
            return NegotiationOutcome.REJECTED, None
        
        site_agent = self.site_agents[offer.to_agent]
        
        context = market_context or {
            "num_competing_sites": len(self.site_agents),
            "enrollment_urgency": 0.5,
        }
        
        outcome, counter = site_agent.receive_offer(offer, self.step, context)
        return outcome, counter
    
    def negotiate_capacity(
        self,
        site_id: str,
        desired_capacity: int,
        budget_per_enrollment: float,
        max_rounds: int = 3,
    ) -> Dict[str, Any]:
        """Run a multi-round negotiation with a site."""
        
        if site_id not in self.site_agents:
            return {
                "success": False,
                "error": f"Site {site_id} not found",
                "final_capacity": 0,
                "final_payment": 0,
            }
        
        result = {
            "success": False,
            "rounds": 0,
            "final_capacity": 0,
            "final_payment": 0,
            "history": [],
        }
        
        current_offer = self.make_offer(
            to_site=site_id,
            capacity_requested=desired_capacity,
            payment_offered=budget_per_enrollment,
        )
        
        for round_num in range(max_rounds):
            result["rounds"] = round_num + 1
            
            outcome, counter = self.submit_offer(current_offer)
            
            result["history"].append({
                "round": round_num + 1,
                "offer": {
                    "capacity": current_offer.offered_capacity,
                    "payment": current_offer.offered_payment,
                },
                "outcome": outcome.value,
            })
            
            if outcome == NegotiationOutcome.ACCEPTED:
                result["success"] = True
                result["final_capacity"] = current_offer.offered_capacity
                result["final_payment"] = current_offer.offered_payment
                break
            
            elif outcome == NegotiationOutcome.COUNTEROFFER and counter:
                # Consider the counteroffer
                if counter.offered_payment <= budget_per_enrollment * 1.3:
                    # Accept counteroffer
                    current_offer = counter
                else:
                    # Make a new offer splitting the difference
                    new_payment = (current_offer.offered_payment + counter.offered_payment) / 2
                    current_offer = self.make_offer(
                        to_site=site_id,
                        capacity_requested=counter.offered_capacity,
                        payment_offered=new_payment,
                    )
            
            elif outcome == NegotiationOutcome.REJECTED:
                break
        
        return result
    
    def get_site_recommendations(
        self,
        desired_capacity: int,
        budget: float,
    ) -> List[Dict[str, Any]]:
        """Get recommendations for which sites to negotiate with."""
        
        recommendations = []
        
        for site_id, agent in self.site_agents.items():
            public_info = agent.get_public_info()
            
            # Score based on available capacity, conversion rate, and relationship
            capacity_score = min(1.0, public_info["available_capacity_estimate"] / max(1, desired_capacity))
            conversion_score = public_info["conversion_rate_estimate"]
            relationship_score = public_info["relationship_score"]
            
            overall_score = (
                capacity_score * 0.4 +
                conversion_score * 0.35 +
                relationship_score * 0.25
            )
            
            recommendations.append({
                "site_id": site_id,
                "score": round(overall_score, 3),
                "available_capacity": public_info["available_capacity_estimate"],
                "conversion_estimate": round(public_info["conversion_rate_estimate"], 3),
                "relationship": round(public_info["relationship_score"], 3),
            })
        
        return sorted(recommendations, key=lambda x: -x["score"])
    
    def step_forward(self) -> None:
        """Advance the negotiation step counter."""
        self.step += 1
    
    def get_market_state(self) -> Dict[str, Any]:
        """Get current market state for strategic decisions."""
        
        total_capacity = sum(
            agent.state.available_capacity()
            for agent in self.site_agents.values()
        )
        
        total_enrolled = sum(
            agent.state.current_enrollment
            for agent in self.site_agents.values()
        )
        
        return {
            "num_sites": len(self.site_agents),
            "total_available_capacity": total_capacity,
            "total_enrolled": total_enrolled,
            "site_summaries": [
                agent.get_public_info()
                for agent in self.site_agents.values()
            ],
        }
    
    def to_dict(self) -> Dict[str, Any]:
        """Serialize negotiator state."""
        return {
            "step": self.step,
            "offer_counter": self.offer_counter,
            "sites": {
                site_id: {
                    "site_id": agent.state.site_id,
                    "true_capacity": agent.state.true_capacity,
                    "current_enrollment": agent.state.current_enrollment,
                    "accepted_capacity": agent.state.accepted_capacity,
                    "relationship_score": agent.history.relationship_score,
                }
                for site_id, agent in self.site_agents.items()
            },
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "MultiAgentNegotiator":
        """Deserialize negotiator state."""
        negotiator = cls()
        negotiator.step = data.get("step", 0)
        negotiator.offer_counter = data.get("offer_counter", 0)
        
        for site_id, site_data in data.get("sites", {}).items():
            negotiator.site_agents[site_id] = SiteAgent(
                site_id=site_id,
                true_capacity=site_data.get("true_capacity", 50),
            )
            negotiator.site_agents[site_id].state.current_enrollment = site_data.get("current_enrollment", 0)
            negotiator.site_agents[site_id].state.accepted_capacity = site_data.get("accepted_capacity", 0)
            negotiator.site_agents[site_id].history.relationship_score = site_data.get("relationship_score", 0.5)
        
        return negotiator
