import pandas as pd
import numpy as np

def compute_anomalies(file_path, threshold_percentile=95):
    print(f"Loading data from {file_path}...\n")
    df = pd.read_csv(file_path)
    
    # Preprocessing
    if 'timestamp' in df.columns:
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df['hour'] = df['timestamp'].dt.hour
        df['day'] = df['timestamp'].dt.day
    else:
        df['hour'] = 0
        df['day'] = 0
        
    global_mean = df['amount'].mean()
    global_std = df['amount'].std()
    if global_std == 0:
        global_std = 1.0
        
    df['z_score_amount'] = np.abs((df['amount'] - global_mean) / global_std)
    
    type_stats = df.groupby('transaction_type')['amount'].agg(['mean', 'std']).reset_index()
    type_stats['std'] = type_stats['std'].fillna(global_std)
    type_stats.loc[type_stats['std'] == 0, 'std'] = global_std
    
    df = df.merge(type_stats, on='transaction_type', suffixes=('', '_type'))
    
    df['z_score_type'] = np.abs((df['amount'] - df['mean']) / df['std'])
    df['unusual_hour_score'] = np.where((df['hour'] >= 0) & (df['hour'] <= 5), 2.0, 0.0)
    
    df['anomaly_score'] = df['z_score_amount'] + df['z_score_type'] + df['unusual_hour_score']
    
    # Flagging criteria: >= 95th percentile
    threshold = np.percentile(df['anomaly_score'], threshold_percentile)
    df['is_flagged'] = df['anomaly_score'] >= threshold
    
    # Report metrics
    num_flagged = df['is_flagged'].sum()
    total_transactions = len(df)
    perc_flagged = (num_flagged / total_transactions) * 100
    
    print(f"--- Flagging Report (Threshold > {threshold:.4f} @ {threshold_percentile}th Percentile) ---")
    print(f"Total Transactions: {total_transactions}")
    print(f"Number of Transactions Flagged: {num_flagged}")
    print(f"Percentage Flagged: {perc_flagged:.2f}%")
    
    print("\n--- Examples of Flagged Transactions ---")
    examples = df[df['is_flagged']].sort_values('anomaly_score', ascending=False).head(10)
    print(examples[['transaction_id', 'amount', 'transaction_type', 'hour', 'anomaly_score']].to_string(index=False))

if __name__ == "__main__":
    file_path = "/Users/ananthchowdary/Desktop/The Truman Show - train/transactions.csv"
    compute_anomalies(file_path)
