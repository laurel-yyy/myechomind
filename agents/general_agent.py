"""General customer-service agent."""

from agents.base_agent import BaseAgent


class GeneralAgent(BaseAgent):
    agent_type = "general"
    role_description = "General Agent for orders, delivery, membership, policy, and customer coordination"
