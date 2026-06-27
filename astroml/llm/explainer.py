"""Fraud Alert Explainer."""
import time
from typing import Any, Dict, List
from .providers.factory import get_llm_provider
from .cache import SemanticCache
from .tracker import global_tracker

class FraudExplainer:
    """Generates explanations for fraud alerts with evidence."""

    def __init__(self):
        self.provider = get_llm_provider()
        self.cache = SemanticCache(ttl=86400) # Cache for 24 hours

    def generate_explanation(self, alert_id: int, account_id: str, pattern: str, score: float, transactions: List[Dict[str, Any]]) -> str:
        """
        Generate an explanation for a fraud alert, citing transactions.
        """
        prompt = self._build_prompt(account_id, pattern, score, transactions)
        
        # Check cache
        cached_response = self.cache.get(prompt)
        if cached_response:
            return cached_response
            
        start_time = time.time()
        
        try:
            response = self.provider.generate(prompt)
            latency_ms = (time.time() - start_time) * 1000.0
            
            # Record usage
            usage = self.provider.get_token_usage()
            global_tracker.record_usage(
                provider_name=self.provider.__class__.__name__.replace("Provider", "").lower(),
                usage=usage,
                latency_ms=latency_ms
            )
            
            # Cache the response
            self.cache.set(prompt, response)
            
            return response
        except Exception as e:
            return f"Error generating explanation: {str(e)}"

    def _build_prompt(self, account_id: str, pattern: str, score: float, transactions: List[Dict[str, Any]]) -> str:
        tx_str = "\n".join([f"- Tx {tx.get('hash')}: {tx.get('amount')} {tx.get('asset_code')} to {tx.get('destination_account')} (Ledger: {tx.get('ledger_sequence')})" for tx in transactions[:5]])
        
        return f"""
        Explain why the following account was flagged for fraud.
        
        Account ID: {account_id}
        Fraud Pattern: {pattern}
        Risk Score: {score:.2f}
        
        Recent Transactions Evidence:
        {tx_str}
        
        Provide a concise explanation for the alert, citing at least 3 transactions as evidence if available.
        """
