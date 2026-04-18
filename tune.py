import pandas as pd
from data_loader import DataLoader
from agents import MemoryAgent, GlobalThreatAgent, SecurityContextAgent, HeuristicAgent, RiskAgent, DecisionAgent

class MockLLMAgent:
    def __init__(self):
        self.call_count = 0
        
    def llm_check(self, prompt, session_id):
        self.call_count += 1
        # Mock assumption for tuning: assume 50% of ambiguities are actual fraud
        return "YES" if self.call_count % 2 == 0 else "NO"

def run_tuning_loop(anomaly_hard, high_val):
    loader = DataLoader()
    context_data = loader.load_all()

    df = pd.read_csv("data/transactions.csv")
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df["hour"] = df["timestamp"].dt.hour

    frauds = []
    
    memory = MemoryAgent()
    global_threat = GlobalThreatAgent()
    sec_context = SecurityContextAgent(context_data)
    heuristic = HeuristicAgent()
    risk = RiskAgent()
    
    decision = DecisionAgent(anomaly_hard=anomaly_hard, high_value_gate=high_val)
    # Monkeypatch the LLMAgent dynamically to intercept API hits
    mock_llm = MockLLMAgent()
    decision.llm_agent = mock_llm

    for _, tx in df.iterrows():
        user = tx["sender_id"]
        profile = memory.get_profile(user)
        
        is_fraudulent = False
        
        if profile:
            intel = sec_context.fetch_intel(tx)
            threat_multiplier = global_threat.check(tx)
            
            anomaly = heuristic.anomaly_score(tx, profile, intel)
            risk_val = risk.risk_score(tx, anomaly, threat_multiplier)
            
            if decision.is_fraud(tx, profile, intel, risk_val, "tuning-session"):
                is_fraudulent = True
                global_threat.register_fraud(tx)
                
        memory.update_profile(tx)
        
        if is_fraudulent:
            frauds.append(tx["transaction_id"])
            
    total = len(df)
    flagged = len(frauds)
    flag_percent = (flagged / total) * 100
    llm_percent = (mock_llm.call_count / total) * 100
    
    return {
        "hard_limit": anomaly_hard,
        "high_val": high_val,
        "flags": flagged,
        "flag_percent": flag_percent,
        "llm_calls": mock_llm.call_count,
        "llm_percent": llm_percent
    }

def main():
    print("--- Starting Auto-Tuner ---")
    configs = [
        {"hard": 5, "high": 10000},
        {"hard": 5, "high": 20000},
        {"hard": 6, "high": 10000},
        {"hard": 6, "high": 20000}
    ]
    
    results = []
    for c in configs:
        res = run_tuning_loop(c["hard"], c["high"])
        results.append(res)
        
    print("\n--- Tuning Results ---")
    best = None
    for r in results:
        print(f"Config: Hard>={r['hard_limit']}, HighVal>={r['high_val']}")
        print(f"  -> Flags: {r['flags']} ({r['flag_percent']:.1f}%) | LLM Usage: {r['llm_percent']:.1f}%")
        
        # Select best: 5-20% flags AND <30% LLM
        if 5 <= r['flag_percent'] <= 28 and r['llm_percent'] < 30:
            if best is None:
                best = r
            # Minimize LLM usage while keeping flags high enough
            elif r['llm_percent'] < best['llm_percent'] and r['flag_percent'] >= 5:
                best = r
                
    if best is not None:
        print("\n🏆 BEST WINNING CONFIGURATION:")
        print(f"  Anomaly Hard Limit: {best['hard_limit']}")
        print(f"  High Value Gate: {best['high_val']}")
    else:
        print("\n⚠️ No configuration strictly hit all bounds simultaneously.")

if __name__ == "__main__":
    main()
