import pandas as pd
import json

class DataLoader:
    def __init__(self, data_path="data"):
        self.data_path = data_path
        self.users = {}
        self.locations = {}
        self.compromise_windows = {}

    def load_all(self):
        print("Loading auxiliary context datasets...")
        
        # 1. Load User Demographics
        try:
            user_df = pd.read_json(f"{self.data_path}/users.json")
            for _, row in user_df.iterrows():
                iban = row.get("iban")
                if pd.notna(iban):
                    self.users[iban] = {
                        "name": f"{row.get('first_name')} {row.get('last_name')}",
                        "salary": row.get("salary"),
                        "job": row.get("job"),
                        "description": row.get("description")
                    }
        except Exception as e:
            print("User init error:", e)

        # 2. Load GPS Trajectories
        try:
            loc_df = pd.read_json(f"{self.data_path}/locations.json")
            for _, row in loc_df.iterrows():
                bt = row["biotag"]
                if bt not in self.locations:
                    self.locations[bt] = []
                self.locations[bt].append({
                    "timestamp": pd.to_datetime(row["timestamp"]),
                    "city": row["city"]
                })
        except Exception as e:
            print("Location init error:", e)

        # 3. Communications (Phishing / ATO Profiling)
        phishing_keywords = ["action required", "urgent", "verify", "password", "secure link"]
        
        try:
            mail_df = pd.read_json(f"{self.data_path}/mails.json")
            for _, row in mail_df.iterrows():
                content = str(row.get("mail", "")).lower()
                if any(kw in content for kw in phishing_keywords):
                    for iban, u in self.users.items():
                        if u["name"].lower() in content:
                            self.compromise_windows[iban] = True
        except Exception:
            pass

        try:
            sms_df = pd.read_json(f"{self.data_path}/sms.json")
            for _, row in sms_df.iterrows():
                content = str(row.get("sms", "")).lower()
                if any(kw in content for kw in phishing_keywords):
                    for iban, u in self.users.items():
                        # SMS usually uses first name greetings
                        if u["name"].split(" ")[0].lower() in content:
                            self.compromise_windows[iban] = True
        except Exception:
            pass

        return {
            "users": self.users,
            "locations": self.locations,
            "compromise": self.compromise_windows
        }
