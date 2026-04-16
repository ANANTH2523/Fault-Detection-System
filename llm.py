from __future__ import annotations

import os
import re
from typing import Any, Dict, Optional

try:
    from langfuse.decorators import observe
except Exception:  # pragma: no cover
    def observe(*_args: Any, **_kwargs: Any):  # type: ignore
        def _decorator(fn):
            return fn

        return _decorator


_FRAUD_RE = re.compile(r"fraud\s*:\s*(yes|no)", flags=re.IGNORECASE)


def _build_langfuse_callback(session_id: str) -> Optional[Any]:
    """
    Build a Langfuse CallbackHandler (LangChain integration).
    Returns None when Langfuse isn't configured or the import fails.
    """
    try:
        # Langfuse's LangChain handler.
        from langfuse.langchain import CallbackHandler  # type: ignore
    except Exception:
        return None

    # If keys are absent, Langfuse shouldn't be used to avoid noisy errors.
    if not os.getenv("LANGFUSE_PUBLIC_KEY") or not os.getenv("LANGFUSE_SECRET_KEY"):
        return None

    try:
        # We keep args minimal for compatibility across Langfuse versions.
        return CallbackHandler(session_id=session_id)  # type: ignore[arg-type]
    except Exception:
        try:
            return CallbackHandler()  # type: ignore[call-arg]
        except Exception:
            return None


@observe(name="llm_analysis")
def llm_analysis(
    profile_summary: str,
    tx_details: Dict[str, Any],
    session_id: str,
    model: Optional[str] = None,
    temperature: float = 0.0,
    callback_handler: Optional[Any] = None,
) -> bool:
    """
    LLM Reasoning Agent (selective):
    Returns True/False, while the model is instructed to output strictly:
      "Fraud: YES" or "Fraud: NO"
    """
    temperature_f = float(temperature)
    if temperature_f < 0.0:
        temperature_f = 0.0
    temperature_f = min(0.1, temperature_f)

    api_key = os.getenv("OPENAI_API_KEY", "")
    model_name = model or os.getenv("OPENAI_MODEL", "gpt-4o-mini")

    system_prompt = (
        "You are a strict fraud detection classifier for financial transactions. "
        'Respond with ONLY one line of text in the exact format: "Fraud: YES" or "Fraud: NO". '
        "Do not include any other text."
    )

    # Keep prompt deterministic and low-latency.
    user_prompt_parts = [
        "User behavior summary:",
        profile_summary.strip() if profile_summary else "user history unavailable",
        "",
        "Transaction details:",
    ]
    for k in (
        "tx_id",
        "amount",
        "type",
        "timestamp",
        "location_id",
        "receiver_id",
        "anomaly_score",
        "risk_score",
    ):
        if k in tx_details:
            user_prompt_parts.append(f"- {k}: {tx_details.get(k)}")
    user_prompt_parts.append("")
    user_prompt_parts.append("Decision:")
    user_prompt_parts.append("Is this transaction fraudulent?")
    user_prompt_parts.append('Answer with exactly: "Fraud: YES" or "Fraud: NO".')

    user_prompt = "\n".join(user_prompt_parts)

    llm_text: str = ""
    if api_key:
        try:
            from langchain_core.messages import SystemMessage, HumanMessage  # type: ignore
            from langchain_openai import ChatOpenAI  # type: ignore

            llm = ChatOpenAI(
                model=model_name,
                temperature=temperature_f,
                max_tokens=5,
                api_key=api_key,
            )
            messages = [SystemMessage(content=system_prompt), HumanMessage(content=user_prompt)]

            # Langfuse callback handler for tracing.
            cb = callback_handler or _build_langfuse_callback(session_id=session_id)
            config: Dict[str, Any] = {}
            if cb is not None:
                config = {"callbacks": [cb]}

            resp = llm.invoke(messages, config=config if config else None)
            # LangChain returns an AIMessage with .content
            llm_text = (getattr(resp, "content", None) or "").strip()
        except Exception:
            llm_text = ""

    if llm_text:
        match = _FRAUD_RE.search(llm_text)
        if match:
            return match.group(1).strip().lower() == "yes"

    # Deterministic fallback heuristic: map based on provided scores.
    try:
        risk_f = float(tx_details.get("risk_score", 0.0) or 0.0)
        anomaly_f = float(tx_details.get("anomaly_score", 0.0) or 0.0)
        amount_f = float(tx_details.get("amount", 0.0) or 0.0)
    except Exception:
        risk_f = 0.0
        anomaly_f = 0.0
        amount_f = 0.0

    if anomaly_f >= 0.65 and risk_f >= 60.0:
        return True
    if risk_f >= 78.0:
        return True

    # Otherwise default to NO.
    return False


