"""Technical reliability agent."""

from agents.base_agent import BaseAgent


class TechnicalAgent(BaseAgent):
    agent_type = "technical"
    role_description = "Technical Reliability Agent for login, errors, crashes, and performance incidents"
