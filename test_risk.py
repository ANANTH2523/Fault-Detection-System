import pandas as pd
from data_loader import DataLoader
from agents import MemoryAgent, GlobalThreatAgent, SecurityContextAgent, HeuristicAgent, RiskAgent
loader = DataLoader()
context_data = loader.load_all()
df = pd.read_csv("data/transactions.csv")
df["timestamp"] = pd.to_datetime(df["timestamp"])
df["hour"] = df["timestamp"].dt.hour
memory = MemoryAgent()
global_threat = GlobalThreatAgent()
sec_context = SecurityContextAgent(context_data)
heuristic = HeuristicAgent()
risk = RiskAgent()
scores = []
for _, tx in df.iterrows():
    user = tx["sender_id"]
    profile = memory.get_profile(user)
    if profile:
        intel = sec_context.fetch_intel(tx)
        threat_multiplier = global_threat.check(tx)
        anomaly = heuristic.anomaly_score(tx, profile, intel)
        rf = risk.risk_score(tx, anomaly, threat_multiplier)
        scores.append(rf)
    memory.update_profile(tx)
s = pd.Series(scores)
print(s.value_counts().sort_index())
