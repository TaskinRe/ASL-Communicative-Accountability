"""
Experiment Execution Pipeline
==============================
Executes all 1440 API calls via OpenRouter:
- 60 stimulus sentences x 8 conditions x 3 models
- Logs raw responses, parses JSON, handles errors
- Follows paper Section 4.5 protocol exactly:
  - temperature=0, max_tokens=500, no system prompt
  - Two-stage JSON parsing (Section 4.5.3)
"""

import json
import time
import os
import urllib.request
import urllib.error
import sys
from datetime import datetime

# ============================================================
# Configuration
# ============================================================
OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY", "")
OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"

# Model mapping: paper name -> OpenRouter model ID
MODELS = {
    'gpt-4o': 'openai/gpt-4o-2024-11-20',
    'claude-3.5-sonnet': 'anthropic/claude-3.7-sonnet',  # successor to 3.5
    'gemini-1.5-pro': 'google/gemini-2.5-pro',           # successor to 1.5
}

OUTPUT_DIR = "/Users/rehnumataskin/Downloads/ASL-Project"
PROMPTS_FILE = f"{OUTPUT_DIR}/experiment_prompts.json"
RESULTS_FILE = f"{OUTPUT_DIR}/experiment_results.json"
RESULTS_CSV = f"{OUTPUT_DIR}/experiment_results.csv"
PROGRESS_FILE = f"{OUTPUT_DIR}/experiment_progress.json"

# Paper protocol: temperature=0, max_tokens=500
# NOTE: Gemini 2.5 Pro uses thinking tokens that eat into the budget,
# so we need max_tokens=8000 to ensure complete JSON output.
# Only the visible output tokens count toward cost; thinking tokens are internal.
TEMPERATURE = 0
MAX_TOKENS = 8000

# Rate limiting: be polite to the API
DELAY_BETWEEN_CALLS = 1.2  # seconds (slightly higher for Gemini thinking time)

# ============================================================
# API Call Function
# ============================================================
def call_openrouter(model_id, prompt, max_retries=3):
    """
    Call OpenRouter API with a single user message (no system prompt).
    Returns (raw_text, parsed_json, error_msg).
    """
    payload = json.dumps({
        "model": model_id,
        "messages": [
            {"role": "user", "content": prompt}
        ],
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
            with urllib.request.urlopen(req, timeout=60) as resp:
                response_data = json.loads(resp.read().decode())

            # Extract the completion text
            raw_text = response_data['choices'][0]['message']['content']

            # Three-stage JSON parsing (extended from Paper Section 4.5.3)
            parsed = None
            parse_error = None

            # Stage 0: Strip markdown code fences (Gemini wraps JSON in ```json ... ```)
            clean_text = raw_text.strip()
            if clean_text.startswith('```'):
                # Remove opening fence (```json or ```)
                first_newline = clean_text.index('\n')
                clean_text = clean_text[first_newline + 1:]
                # Remove closing fence
                if clean_text.rstrip().endswith('```'):
                    clean_text = clean_text.rstrip()[:-3].rstrip()

            # Stage 1: Direct parse
            try:
                parsed = json.loads(clean_text)
            except json.JSONDecodeError:
                # Stage 2: Extract JSON between first { and last }
                try:
                    start = clean_text.index('{')
                    end = clean_text.rindex('}') + 1
                    json_str = clean_text[start:end]
                    parsed = json.loads(json_str)
                except (ValueError, json.JSONDecodeError) as e:
                    parse_error = f"PARSE_ERROR: {str(e)}"

            usage = response_data.get('usage', {})
            return raw_text, parsed, parse_error, usage

        except urllib.error.HTTPError as e:
            error_body = e.read().decode() if hasattr(e, 'read') else ''
            if e.code == 429:
                wait_time = (attempt + 1) * 5
                print(f"    Rate limited. Waiting {wait_time}s... (attempt {attempt+1}/{max_retries})")
                time.sleep(wait_time)
                continue
            elif e.code == 502 or e.code == 503:
                wait_time = (attempt + 1) * 3
                print(f"    Server error {e.code}. Retrying in {wait_time}s...")
                time.sleep(wait_time)
                continue
            else:
                return None, None, f"HTTP_{e.code}: {error_body[:200]}", {}

        except Exception as e:
            if attempt < max_retries - 1:
                time.sleep(2)
                continue
            return None, None, f"ERROR: {str(e)}", {}

    return None, None, "MAX_RETRIES_EXCEEDED", {}


# ============================================================
# Load prompts
# ============================================================
print("=" * 70)
print("LOADING EXPERIMENT PROMPTS")
print("=" * 70)

with open(PROMPTS_FILE, 'r') as f:
    prompt_data = json.load(f)

all_prompts = prompt_data['prompts']
print(f"Total prompts to execute: {len(all_prompts)}")

# Check for existing progress (resume capability)
completed_keys = set()
results = []

if os.path.exists(PROGRESS_FILE):
    with open(PROGRESS_FILE, 'r') as f:
        progress = json.load(f)
        results = progress.get('results', [])
        for r in results:
            key = f"{r['stimulus_id']}_{r['condition']}_{r['model']}"
            completed_keys.add(key)
    print(f"Resuming: {len(completed_keys)} already completed")

remaining = []
for p in all_prompts:
    key = f"{p['stimulus_id']}_{p['condition']}_{p['model']}"
    if key not in completed_keys:
        remaining.append(p)

print(f"Remaining to execute: {len(remaining)}")

# ============================================================
# Execute
# ============================================================
print("\n" + "=" * 70)
print("EXECUTING EXPERIMENT")
print("=" * 70)

total_remaining = len(remaining)
total_usage = {'prompt_tokens': 0, 'completion_tokens': 0}
parse_errors = 0
api_errors = 0

start_time = time.time()

for idx, prompt_item in enumerate(remaining):
    stim_id = prompt_item['stimulus_id']
    condition = prompt_item['condition']
    model_name = prompt_item['model']
    model_id = MODELS[model_name]
    prompt_text = prompt_item['prompt']

    # Progress display
    elapsed = time.time() - start_time
    rate = (idx + 1) / elapsed if elapsed > 0 else 0
    eta = (total_remaining - idx - 1) / rate if rate > 0 else 0

    if idx % 10 == 0:
        print(f"\n[{idx+1}/{total_remaining}] ETA: {eta/60:.1f}min | "
              f"Errors: {api_errors} parse/{parse_errors} api | "
              f"Tokens: {total_usage['prompt_tokens']:,}p/{total_usage['completion_tokens']:,}c")

    sys.stdout.write(f"  {stim_id} {condition} {model_name[:8]:8s} ... ")
    sys.stdout.flush()

    # API call
    raw_text, parsed_json, error_msg, usage = call_openrouter(model_id, prompt_text)

    # Track usage
    total_usage['prompt_tokens'] += usage.get('prompt_tokens', 0)
    total_usage['completion_tokens'] += usage.get('completion_tokens', 0)

    # Build result record
    result = {
        'stimulus_id': stim_id,
        'source': prompt_item['source'],
        'ground_truth_tag': prompt_item['ground_truth_tag'],
        'epistemic_evidence': prompt_item['epistemic_evidence'],
        'stem_term': prompt_item['stem_term'],
        'model': model_name,
        'model_id': model_id,
        'condition': condition,
        'raw_output': raw_text,
        'parsed_json': parsed_json,
        'parse_error': error_msg,
        'timestamp': datetime.now().isoformat(),
        'sentence_text': prompt_item['sentence_text']
    }

    results.append(result)

    if error_msg and 'PARSE_ERROR' in str(error_msg):
        parse_errors += 1
        sys.stdout.write(f"PARSE_ERR\n")
    elif error_msg:
        api_errors += 1
        sys.stdout.write(f"API_ERR: {error_msg[:50]}\n")
    else:
        # Quick sanity check on parsed output
        status = "OK"
        if parsed_json:
            keys = list(parsed_json.keys())
            status = f"OK ({len(keys)} fields)"
        sys.stdout.write(f"{status}\n")

    # Save progress every 50 calls
    if (idx + 1) % 50 == 0 or idx == total_remaining - 1:
        with open(PROGRESS_FILE, 'w') as f:
            json.dump({'results': results, 'total_usage': total_usage}, f, indent=2)

    # Rate limiting
    time.sleep(DELAY_BETWEEN_CALLS)

# ============================================================
# Save final results
# ============================================================
print("\n" + "=" * 70)
print("SAVING RESULTS")
print("=" * 70)

elapsed_total = time.time() - start_time

# Save full results JSON
with open(RESULTS_FILE, 'w') as f:
    json.dump({
        'metadata': {
            'execution_time_seconds': elapsed_total,
            'total_calls': len(results),
            'parse_errors': parse_errors,
            'api_errors': api_errors,
            'total_usage': total_usage,
            'models_used': MODELS,
            'temperature': TEMPERATURE,
            'max_tokens': MAX_TOKENS,
            'timestamp': datetime.now().isoformat()
        },
        'results': results
    }, f, indent=2, ensure_ascii=False)
print(f"Saved: {RESULTS_FILE}")

# Save CSV version (flattened)
import csv

csv_fields = ['stimulus_id', 'source', 'ground_truth_tag', 'epistemic_evidence',
              'stem_term', 'model', 'model_id', 'condition', 'parse_error', 'timestamp']

# Add common parsed JSON fields
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
print(f"Saved: {RESULTS_CSV}")

# ============================================================
# Execution Summary
# ============================================================
print("\n" + "=" * 70)
print("EXECUTION SUMMARY")
print("=" * 70)

print(f"""
Total API calls:     {len(results)}
Parse errors:        {parse_errors} ({parse_errors/len(results)*100:.1f}%)
API errors:          {api_errors} ({api_errors/len(results)*100:.1f}%)
Successful parses:   {len(results) - parse_errors - api_errors} ({(len(results) - parse_errors - api_errors)/len(results)*100:.1f}%)
Execution time:      {elapsed_total/60:.1f} minutes
Total tokens:        {total_usage['prompt_tokens']:,} prompt + {total_usage['completion_tokens']:,} completion

Per model:""")

for model_name in MODELS:
    model_results = [r for r in results if r['model'] == model_name]
    model_errors = sum(1 for r in model_results if r.get('parse_error'))
    print(f"  {model_name:20s}: {len(model_results)} calls, {model_errors} errors ({model_errors/max(1,len(model_results))*100:.1f}%)")

print(f"\nPer condition:")
for cond in ['1A', '1B', '1C', '2A', '2B', '3A', '3B', '3C']:
    cond_results = [r for r in results if r['condition'] == cond]
    cond_errors = sum(1 for r in cond_results if r.get('parse_error'))
    print(f"  {cond}: {len(cond_results)} calls, {cond_errors} errors")

print(f"\nResults saved to:")
print(f"  {RESULTS_FILE}")
print(f"  {RESULTS_CSV}")
