"""
FAST Experiment Execution Pipeline (Parallel + Fast Models)
============================================================
- Uses GPT-4o-mini, Claude 3.5 Haiku, Gemini 2.0 Flash
- Parallel execution (3 concurrent requests)
- Estimated: ~8-12 minutes for all 1440 calls
"""

import json
import time
import os
import urllib.request
import urllib.error
import sys
import concurrent.futures
from datetime import datetime
import threading

# ============================================================
# Configuration
# ============================================================
OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY", "")
if not OPENROUTER_API_KEY:
    print("ERROR: Set your OPENROUTER_API_KEY environment variable first.")
    print("  export OPENROUTER_API_KEY='your-key-here'")
    sys.exit(1)
OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"

# Fast models - still instruction-tuned, same providers, much faster
MODELS = {
    'gpt-4o': 'openai/gpt-4o-mini',              # Fast OpenAI
    'claude-3.5-sonnet': 'anthropic/claude-3.5-haiku',  # Fast Anthropic
    'gemini-1.5-pro': 'google/gemini-2.0-flash-001',    # Fast Google (no thinking)
}

OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))
PROMPTS_FILE = f"{OUTPUT_DIR}/experiment_prompts.json"
RESULTS_FILE = f"{OUTPUT_DIR}/experiment_results.json"
RESULTS_CSV = f"{OUTPUT_DIR}/experiment_results.csv"

TEMPERATURE = 0
MAX_TOKENS = 500  # No thinking mode = 500 is plenty

# Parallelism
MAX_WORKERS = 6  # 6 concurrent requests
RATE_LIMIT_DELAY = 0.2  # seconds between launches

# Thread-safe counters
lock = threading.Lock()
completed = 0
parse_errors = 0
api_errors = 0
total_tokens = {'prompt': 0, 'completion': 0}

# ============================================================
# API Call Function
# ============================================================
def call_openrouter(model_id, prompt, max_retries=3):
    """Call OpenRouter with retry logic."""
    payload = json.dumps({
        "model": model_id,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": TEMPERATURE,
        "max_tokens": MAX_TOKENS,
    }).encode('utf-8')

    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://github.com/ASL-Project",
        "X-Title": "ASL Communicative Accountability Probe"
    }

    for attempt in range(max_retries):
        try:
            req = urllib.request.Request(OPENROUTER_URL, data=payload, headers=headers, method='POST')
            with urllib.request.urlopen(req, timeout=45) as resp:
                response_data = json.loads(resp.read().decode())

            raw_text = response_data['choices'][0]['message']['content']
            usage = response_data.get('usage', {})

            # Three-stage parsing
            parsed = None
            parse_error = None

            # Stage 0: strip markdown fences
            clean_text = raw_text.strip()
            if clean_text.startswith('```'):
                try:
                    first_nl = clean_text.index('\n')
                    clean_text = clean_text[first_nl + 1:]
                    if clean_text.rstrip().endswith('```'):
                        clean_text = clean_text.rstrip()[:-3].rstrip()
                except ValueError:
                    pass

            # Stage 1: direct parse
            try:
                parsed = json.loads(clean_text)
            except json.JSONDecodeError:
                # Stage 2: extract between { and }
                try:
                    start = clean_text.index('{')
                    end = clean_text.rindex('}') + 1
                    parsed = json.loads(clean_text[start:end])
                except (ValueError, json.JSONDecodeError) as e:
                    parse_error = f"PARSE_ERROR: {str(e)[:100]}"

            return raw_text, parsed, parse_error, usage

        except urllib.error.HTTPError as e:
            if e.code == 429:
                time.sleep((attempt + 1) * 3)
                continue
            elif e.code in (502, 503, 504):
                time.sleep((attempt + 1) * 2)
                continue
            else:
                body = e.read().decode()[:200] if hasattr(e, 'read') else ''
                return None, None, f"HTTP_{e.code}: {body}", {}
        except Exception as e:
            if attempt < max_retries - 1:
                time.sleep(1)
                continue
            return None, None, f"ERROR: {str(e)[:100]}", {}

    return None, None, "MAX_RETRIES_EXCEEDED", {}


def process_prompt(prompt_item):
    """Process a single prompt item. Called in parallel."""
    global completed, parse_errors, api_errors, total_tokens

    model_name = prompt_item['model']
    model_id = MODELS[model_name]
    prompt_text = prompt_item['prompt']

    raw_text, parsed_json, error_msg, usage = call_openrouter(model_id, prompt_text)

    # Update counters
    with lock:
        completed += 1
        if usage:
            total_tokens['prompt'] += usage.get('prompt_tokens', 0)
            total_tokens['completion'] += usage.get('completion_tokens', 0)
        if error_msg and 'PARSE_ERROR' in str(error_msg):
            parse_errors += 1
        elif error_msg:
            api_errors += 1

    result = {
        'stimulus_id': prompt_item['stimulus_id'],
        'source': prompt_item['source'],
        'ground_truth_tag': prompt_item['ground_truth_tag'],
        'epistemic_evidence': prompt_item['epistemic_evidence'],
        'stem_term': prompt_item['stem_term'],
        'model': model_name,
        'model_id': model_id,
        'condition': prompt_item['condition'],
        'raw_output': raw_text,
        'parsed_json': parsed_json,
        'parse_error': error_msg,
        'timestamp': datetime.now().isoformat(),
        'sentence_text': prompt_item['sentence_text']
    }

    return result


# ============================================================
# Main Execution
# ============================================================
print("=" * 70)
print("FAST EXPERIMENT EXECUTION")
print("=" * 70)
print(f"Models: {json.dumps(MODELS, indent=2)}")
print(f"Parallel workers: {MAX_WORKERS}")
print(f"Temperature: {TEMPERATURE} | Max tokens: {MAX_TOKENS}")
print()

# Load prompts
with open(PROMPTS_FILE, 'r') as f:
    prompt_data = json.load(f)

all_prompts = prompt_data['prompts']
total_prompts = len(all_prompts)
print(f"Total prompts: {total_prompts}")

# Execute in parallel
start_time = time.time()
results = []

print(f"\nStarting parallel execution at {datetime.now().strftime('%H:%M:%S')}...")
print("-" * 70)

# Progress reporter
def report_progress():
    elapsed = time.time() - start_time
    rate = completed / elapsed if elapsed > 0 else 0
    remaining = total_prompts - completed
    eta = remaining / rate if rate > 0 else 0
    print(f"  [{completed}/{total_prompts}] {rate:.1f} calls/sec | "
          f"ETA: {eta/60:.1f}min | "
          f"Errors: {parse_errors}p/{api_errors}a | "
          f"Tokens: {total_tokens['prompt']:,}+{total_tokens['completion']:,}")

with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
    # Submit all tasks with small stagger to avoid burst
    futures = []
    for i, prompt_item in enumerate(all_prompts):
        future = executor.submit(process_prompt, prompt_item)
        futures.append(future)
        # Small stagger to avoid rate limits
        if i % MAX_WORKERS == 0 and i > 0:
            time.sleep(RATE_LIMIT_DELAY)

    # Collect results with progress
    last_report = 0
    for future in concurrent.futures.as_completed(futures):
        result = future.result()
        results.append(result)

        # Report every 100 completions
        if completed - last_report >= 100:
            report_progress()
            last_report = completed

# Final progress
report_progress()

elapsed_total = time.time() - start_time
print(f"\n{'=' * 70}")
print(f"EXECUTION COMPLETE in {elapsed_total:.1f}s ({elapsed_total/60:.1f} min)")
print(f"{'=' * 70}")

# ============================================================
# Save Results
# ============================================================
print("\nSaving results...")

# Sort results by stimulus_id and condition for consistent ordering
results.sort(key=lambda r: (r['stimulus_id'], r['condition'], r['model']))

# Full JSON
with open(RESULTS_FILE, 'w') as f:
    json.dump({
        'metadata': {
            'execution_time_seconds': elapsed_total,
            'total_calls': len(results),
            'parse_errors': parse_errors,
            'api_errors': api_errors,
            'total_tokens': total_tokens,
            'models_used': MODELS,
            'temperature': TEMPERATURE,
            'max_tokens': MAX_TOKENS,
            'max_workers': MAX_WORKERS,
            'timestamp': datetime.now().isoformat()
        },
        'results': results
    }, f, indent=2, ensure_ascii=False)
print(f"  {RESULTS_FILE}")

# CSV
import csv
csv_fields = ['stimulus_id', 'source', 'ground_truth_tag', 'epistemic_evidence',
              'stem_term', 'model', 'model_id', 'condition', 'parse_error', 'timestamp']
json_fields = ['reformulation', 'certainty_level', 'epistemic_signal',
               'ASL_gloss', 'jargon_strategy', 'expansion_reasoning',
               'output', 'whose_perspective', 'accountability_note',
               'epistemic_preservation']

with open(RESULTS_CSV, 'w', newline='', encoding='utf-8') as f:
    writer = csv.DictWriter(f, fieldnames=csv_fields + json_fields, extrasaction='ignore')
    writer.writeheader()
    for r in results:
        row = {k: r.get(k, '') for k in csv_fields}
        if r.get('parsed_json'):
            for jf in json_fields:
                row[jf] = r['parsed_json'].get(jf, '')
        writer.writerow(row)
print(f"  {RESULTS_CSV}")

# ============================================================
# Summary Statistics
# ============================================================
print(f"\n{'=' * 70}")
print("EXECUTION SUMMARY")
print(f"{'=' * 70}")

successful = len(results) - parse_errors - api_errors
print(f"""
  Total calls:       {len(results)}
  Successful:        {successful} ({successful/len(results)*100:.1f}%)
  Parse errors:      {parse_errors}
  API errors:        {api_errors}
  Time:              {elapsed_total/60:.1f} minutes
  Rate:              {len(results)/elapsed_total:.1f} calls/second
  Tokens:            {total_tokens['prompt']:,} prompt + {total_tokens['completion']:,} completion
""")

# Per model breakdown
print("Per model:")
for model_name, model_id in MODELS.items():
    model_results = [r for r in results if r['model'] == model_name]
    model_errors = sum(1 for r in model_results if r.get('parse_error'))
    model_success = len(model_results) - model_errors
    print(f"  {model_name:20s} ({model_id})")
    print(f"    {model_success}/{len(model_results)} success ({model_success/max(1,len(model_results))*100:.0f}%)")

print("\nPer condition:")
for cond in ['1A', '1B', '1C', '2A', '2B', '3A', '3B', '3C']:
    cond_results = [r for r in results if r['condition'] == cond]
    cond_errors = sum(1 for r in cond_results if r.get('parse_error'))
    print(f"  {cond}: {len(cond_results)} calls, {cond_errors} errors")

# Cost estimate
cost_prompt = total_tokens['prompt'] / 1_000_000 * 0.15  # approx avg of fast models
cost_completion = total_tokens['completion'] / 1_000_000 * 0.60
print(f"\nEstimated cost: ${cost_prompt + cost_completion:.3f}")
print(f"\nDone! Results ready for metric computation.")
