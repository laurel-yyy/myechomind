"""Intent taxonomy: 18 fine-grained intents grouped into 4 buckets.

Downstream modules import `INTENT_CATALOG` and `get_spec()` so the source of
truth stays here. Groups drive Agent routing (general -> GeneralAgent,
technical -> TechnicalAgent, billing -> BillingAgent, escalation -> handoff).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

Urgency = Literal["low", "medium", "high"]
IntentGroup = Literal["general", "technical", "billing", "escalation"]


@dataclass(frozen=True)
class IntentSpec:
    intent: str
    group: IntentGroup
    urgency_bias: Urgency
    keywords: tuple[str, ...] = field(default_factory=tuple)
    examples: tuple[str, ...] = field(default_factory=tuple)
    description: str = ""

    def to_candidate(self) -> dict[str, object]:
        """Shape expected by LLMClient.classify_intent."""
        return {
            "intent": self.intent,
            "group": self.group,
            "keywords": list(self.keywords),
            "description": self.description,
        }


INTENT_CATALOG: tuple[IntentSpec, ...] = (
    # ------------------------ general ------------------------
    IntentSpec(
        intent="greeting", group="general", urgency_bias="low",
        keywords=("hi", "hello", "hey", "good morning", "good evening"),
        examples=("hi", "hello there", "good morning, I have a question"),
        description="user greeting or opening pleasantries",
    ),
    IntentSpec(
        intent="farewell", group="general", urgency_bias="low",
        keywords=("bye", "goodbye", "thanks", "thank you", "see you"),
        examples=("thanks, goodbye", "see you later", "ok, thanks"),
        description="user closing the conversation",
    ),
    IntentSpec(
        intent="feedback", group="general", urgency_bias="low",
        keywords=("feedback", "suggestion", "review", "comment"),
        examples=("I have some feedback about your service", "a small suggestion"),
        description="user leaving feedback or suggestions",
    ),
    IntentSpec(
        intent="unknown", group="general", urgency_bias="medium",
        keywords=(),
        examples=("?", "..."),
        description="cannot be confidently mapped to any specific intent",
    ),
    IntentSpec(
        intent="order_status", group="general", urgency_bias="medium",
        keywords=("order", "status", "order id", "purchase", "my order"),
        examples=("check my order", "what is the status of order ABC123456?"),
        description="user asks about the status of a specific order",
    ),
    IntentSpec(
        intent="logistics", group="general", urgency_bias="medium",
        keywords=("shipping", "delivery", "tracking", "shipment", "package", "arrive"),
        examples=("when will my package arrive?", "how often does tracking update?"),
        description="user asks about shipping, delivery, or tracking",
    ),
    IntentSpec(
        intent="membership", group="general", urgency_bias="low",
        keywords=("membership", "member", "vip", "tier", "upgrade plan", "perks"),
        examples=("how do I upgrade my membership?", "what are the vip perks?"),
        description="user asks about membership tiers, perks, or upgrades",
    ),
    IntentSpec(
        intent="policy", group="general", urgency_bias="low",
        keywords=("policy", "rule", "terms", "return policy", "refund policy"),
        examples=("what's your return policy?", "explain the refund rules"),
        description="user asks about policies, terms, or rules",
    ),
    # ------------------------ technical ------------------------
    IntentSpec(
        intent="login_failure", group="technical", urgency_bias="high",
        keywords=("login", "sign in", "log in", "401", "cannot log", "unable to login"),
        examples=("I keep getting 401 when logging in", "cannot log in, keeps failing"),
        description="user cannot log in or authentication fails",
    ),
    IntentSpec(
        intent="crash", group="technical", urgency_bias="high",
        keywords=("crash", "freeze", "hang", "white screen", "blank screen"),
        examples=("the app crashes every time I open it", "browser froze on checkout"),
        description="the application crashes, freezes, or shows a blank screen",
    ),
    IntentSpec(
        intent="error_code", group="technical", urgency_bias="high",
        keywords=("error", "500", "404", "error code", "exception", "stack trace"),
        examples=("I see error 500 on the checkout page", "getting a 404 on the profile page"),
        description="user encounters a specific error code or exception",
    ),
    IntentSpec(
        intent="performance", group="technical", urgency_bias="medium",
        keywords=("slow", "lag", "laggy", "latency", "sluggish"),
        examples=("the site is very slow today", "the app feels laggy"),
        description="performance issues: slowness, lag, latency",
    ),
    # ------------------------ billing ------------------------
    IntentSpec(
        intent="refund", group="billing", urgency_bias="medium",
        keywords=("refund", "money back", "return money", "reimburse"),
        examples=("I want a refund for my last order", "please refund me"),
        description="user requests a refund",
    ),
    IntentSpec(
        intent="payment_issue", group="billing", urgency_bias="high",
        keywords=("payment", "pay", "payment failed", "cannot pay", "charge failed"),
        examples=("my payment keeps failing", "cannot complete the payment"),
        description="user has trouble making a payment",
    ),
    IntentSpec(
        intent="invoice", group="billing", urgency_bias="low",
        keywords=("invoice", "receipt", "billing statement", "issue an invoice"),
        examples=("please issue an invoice for order ABC", "resend the receipt"),
        description="user asks for or has issues with an invoice",
    ),
    IntentSpec(
        intent="subscription", group="billing", urgency_bias="medium",
        keywords=("subscription", "renew", "cancel", "unsubscribe", "auto-renew"),
        examples=("I want to cancel my subscription", "turn off auto-renew"),
        description="user asks about subscription: renew, cancel, upgrade, downgrade",
    ),
    IntentSpec(
        intent="duplicate_charge", group="billing", urgency_bias="high",
        keywords=("duplicate", "double", "twice", "charged twice", "double charged"),
        examples=("I was charged twice for the same order", "duplicate charge on my card"),
        description="user was charged multiple times for the same purchase",
    ),
    # ------------------------ escalation ------------------------
    IntentSpec(
        intent="human_handoff", group="escalation", urgency_bias="high",
        keywords=("human", "agent", "transfer", "real person", "complaint", "escalate"),
        examples=("can I talk to a human?", "transfer me to a real person", "I want to file a complaint"),
        description="user requests a human agent, complaint escalation, or handoff",
    ),
)


_BY_INTENT: dict[str, IntentSpec] = {s.intent: s for s in INTENT_CATALOG}


def get_spec(intent: str) -> IntentSpec:
    return _BY_INTENT.get(intent, _BY_INTENT["unknown"])


def all_intents() -> list[str]:
    return [s.intent for s in INTENT_CATALOG]


def all_groups() -> list[str]:
    return sorted({s.group for s in INTENT_CATALOG})
