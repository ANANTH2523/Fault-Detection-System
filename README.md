# Fraud Detection (Multi-Agent, Langfuse, Selective LLM Escalation)

## How to run

1. Install dependencies:
   - `pip install -r requirements.txt`
2. Run:
   - `python main.py`
   - Or specify an input file:
     - `python main.py --transactions ./data/transactions.csv`
     - (backward-compatible) `python main.py --data ./data/transactions.csv`
3. Output:
   - Default: `output/output.txt`
   - Override: `python main.py --transactions ./data/transactions.csv --output ./output.txt`

The script writes fraudulent transaction IDs to `output/output.txt` (one ID per line, no extra text).

## Input format

Supports CSV or JSON containing at least:
- transaction ID (`transaction_id` / `tx_id` / `id`)
- user ID (`user_id` / `customer_id` / `account_id`)
- amount (`amount`)

If optional columns exist, the system will use them:
- timestamp (`timestamp`, `time`, `created_at`, `date`)
- latitude/longitude (`lat`/`lon`, `lng`)
- location (`location`, `location_id`, `city`, `state`, `country`)

## Langfuse + LLM

Langfuse instrumentation is enabled via:
- `LANGFUSE_PUBLIC_KEY`
- `LANGFUSE_SECRET_KEY`
- `LANGFUSE_HOST` (optional; defaults to `https://cloud.langfuse.com`)

LLM escalation uses OpenAI via LangChain if:
- `OPENAI_API_KEY` is set

The code degrades gracefully (deterministic fallback) if API credentials are missing, while still respecting the required prompt/output contract.

