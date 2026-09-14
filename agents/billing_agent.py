"""Billing and compliance agent."""

from agents.base_agent import BaseAgent


class BillingAgent(BaseAgent):
    agent_type = "billing"
    role_description = "Billing and Compliance Agent for refunds, invoices, payments, subscriptions, and duplicate charges"
