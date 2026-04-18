import pandas as pd

def process_transactions(file_path):
    print(f"Loading data from {file_path}...\n")
    df = pd.read_csv(file_path)
    
    print("--- First 5 Rows ---")
    print(df.head())
    print("\n")
    
    print("--- All Columns ---")
    print(df.columns.tolist())
    print("\n")
    
    # Convert timestamp to datetime
    if 'timestamp' in df.columns:
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        # Extract hour and day
        df['hour'] = df['timestamp'].dt.hour
        df['day'] = df['timestamp'].dt.day
        
        print("--- After converting timestamp and extracting hour, day ---")
        print(df[['timestamp', 'hour', 'day']].head())
    else:
        print("Column 'timestamp' not found.")

if __name__ == "__main__":
    file_path = "/Users/ananthchowdary/Desktop/The Truman Show - train/transactions.csv"
    process_transactions(file_path)
