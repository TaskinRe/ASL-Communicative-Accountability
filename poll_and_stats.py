import os
"""
PoLL Judge + Self-Report Validation + Statistical Analysis
===========================================================
All in one fast script:
1. PoLL rotating judge (Claude judges GPT, GPT judges Gemini, Gemini judges Claude)
2. Self-report validation (does certainty_level match actual reformulation?)
3. Full statistical analysis with effect sizes and significance tests
"""

import json
import time
import urllib.request
import urllib.error
import concurrent.futures
import threading
import re
from collections import defaultdict
from datetime import datetime
import math

# ============================================================
# Config
# ============================================================
API_KEY = os.environ.get("OPENROUTER_API_KEY", "")
if not API_KEY:
    print("ERROR: Set OPENROUTER_API_KEY environment variable.")
    import sys; sys.exit(1)
URL = "https://openrouter.ai/api/v1/chat/completions"
OUTPUT_DIR = "os.path.dirname(os.path.abspath(__file__))"

# PoLL rotation: no model judges itself
POLL_ROTATION = {
    'gpt-4o': 'anthropic/claude-3.5-haiku',       # Claude judges GPT
    'claude-3.5-sonnet': 'google/gemini-2.0-flash-001',  # Gemini judges Claude
    'gemini-1.5-pro': 'openai/gpt-4o-mini',       # GPT judges Gemini
}

MAX_WORKERS = 6
lock = threading.Lock()
completed_poll = 0

# ============================================================
# Helper: API call
# ============================================================
def call_api(model_id, prompt, max_retries=3):
    payload = json.dumps({
        "model": model_id,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0,
        "max_tokens": 500,
    }).encode('utf-8')
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://github.com/ASL-Project",
        "X-Title": "ASL PoLL Judge"
    }
    for attempt in range(max_retries):
        try:
            req = urllib.request.Request(URL, data=payload, headers=headers, method='POST')
            with urllib.request.urlopen(req, timeout=45) as resp:
                data = json.loads(resp.read().decode())
            raw = data['choices'][0]['message']['content']
            # Parse JSON
            clean = raw.strip()
            if clean.startswith('```'):
                try:
                    clean = clean[clean.index('\n')+1:]
                    if clean.rstrip().endswith('```'):
                        clean = clean.rstrip()[:-3].rstrip()
                except: pass
            try:
                return json.loads(clean), None
            except:
                try:
                    s = clean.index('{')
                    e = clean.rindex('}') + 1
                    return json.loads(clean[s:e]), None
                except Exception as ex:
                    return None, str(ex)
        except urllib.error.HTTPError as e:
            if e.code == 429:
                time.sleep((attempt+1)*3)
            else:
                return None, f"HTTP_{e.code}"
        except Exception as e:
            if attempt < max_retries-1:
                time.sleep(1)
            else:
                return None, str(e)
    return None, "MAX_RETRIES"

# ============================================================
# Load experiment results
# ============================================================
print("=" * 70)
print("LOADING RESULTS")
print("=" * 70)

with open(f"{OUTPUT_DIR}/experiment_results.json", 'r') as f:
    exp_data = json.load(f)
results = exp_data['results']
valid = [r for r in results if r.get('parsed_json') and not r.get('parse_error')]
print(f"Valid results: {len(valid)}")

# ============================================================
# PART 1: PoLL JUDGE SCORING (Dimension 3)
# ============================================================
print("\n" + "=" * 70)
print("PART 1: PoLL JUDGE SCORING")
print("=" * 70)

# Get Dimension 3 results (conditions 3B and 3C only -- 3A has no accountability)
dim3_results = [r for r in valid if r['condition'] in ('3B', '3C')]
print(f"Dimension 3 outputs to judge: {len(dim3_results)}")

# Build judge prompts per Appendix B
def make_judge_prompt(r):
    pj = r['parsed_json']
    model_output = json.dumps(pj, indent=2)
    gt_tag = r['ground_truth_tag']
    original = r['sentence_text']
    
    return f'''You are evaluating whether an AI system demonstrated communicative accountability in its output during an academic interpreting task.

Original sentence: "{original}"
Ground truth epistemic tag: "{gt_tag}"
System output: {model_output}

Rate on three dimensions using this scale:
1 = poor (criterion not met)
2 = adequate (criterion partially met)
3 = good (criterion clearly met)

Dimensions:
1. Epistemic faithfulness: did the output preserve the speaker's certainty level without adding or removing hedging?
2. Audience adaptation: did the output show genuine awareness of the audience's linguistic and accessibility needs?
3. Role accountability: did the output reflect genuine awareness of intermediary responsibility to both speaker and audience?

Output only valid JSON with no other text:
{{"epistemic_faithfulness": int, "audience_adaptation": int, "role_accountability": int, "reasoning": "brief explanation"}}'''


def judge_one(item):
    global completed_poll
    r, judge_model = item
    prompt = make_judge_prompt(r)
    parsed, err = call_api(judge_model, prompt)
    with lock:
        completed_poll += 1
        if completed_poll % 50 == 0:
            print(f"  PoLL progress: {completed_poll}/{len(dim3_results)}")
    return {
        'stimulus_id': r['stimulus_id'],
        'condition': r['condition'],
        'target_model': r['model'],
        'judge_model': judge_model,
        'scores': parsed,
        'error': err
    }

# Prepare judge tasks
judge_tasks = []
for r in dim3_results:
    judge_model = POLL_ROTATION[r['model']]
    judge_tasks.append((r, judge_model))

print(f"Running {len(judge_tasks)} PoLL judge calls in parallel...")
start = time.time()

poll_results = []
with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
    futures = []
    for i, task in enumerate(judge_tasks):
        futures.append(executor.submit(judge_one, task))
        if i % MAX_WORKERS == 0 and i > 0:
            time.sleep(0.2)
    for f in concurrent.futures.as_completed(futures):
        poll_results.append(f.result())

elapsed = time.time() - start
errors = sum(1 for p in poll_results if p['error'])
print(f"PoLL complete: {len(poll_results)} judged in {elapsed:.0f}s ({errors} errors)")

# Compute PoLL-based ASR
print("\n--- PoLL Judge Scores (mean per model per condition) ---")
print(f"{'Model':<22} {'Cond':<6} {'Epist.Faith':>12} {'Aud.Adapt':>12} {'Role.Acc':>12} {'N':>5}")
print("-" * 70)

model_display = {'gpt-4o': 'GPT-4o-mini', 'claude-3.5-sonnet': 'Claude 3.5 Haiku', 'gemini-1.5-pro': 'Gemini 2.0 Flash'}
poll_scores = defaultdict(lambda: defaultdict(list))

for p in poll_results:
    if p['scores']:
        key = (p['target_model'], p['condition'])
        poll_scores[key]['epistemic_faithfulness'].append(p['scores'].get('epistemic_faithfulness', 0))
        poll_scores[key]['audience_adaptation'].append(p['scores'].get('audience_adaptation', 0))
        poll_scores[key]['role_accountability'].append(p['scores'].get('role_accountability', 0))

for model in ['gpt-4o', 'claude-3.5-sonnet', 'gemini-1.5-pro']:
    for cond in ['3B', '3C']:
        key = (model, cond)
        scores = poll_scores[key]
        n = len(scores.get('epistemic_faithfulness', []))
        if n > 0:
            ef = sum(scores['epistemic_faithfulness']) / n
            aa = sum(scores['audience_adaptation']) / n
            ra = sum(scores['role_accountability']) / n
            print(f"{model_display[model]:<22} {cond:<6} {ef:>12.2f} {aa:>12.2f} {ra:>12.2f} {n:>5}")

# PoLL-based ASR (role_accountability >= 2 counts as substantive)
print("\n--- PoLL-based ASR (role_accountability >= 2) ---")
poll_asr = {}
for model in ['gpt-4o', 'claude-3.5-sonnet', 'gemini-1.5-pro']:
    poll_asr[model] = {}
    for cond in ['3B', '3C']:
        key = (model, cond)
        scores = poll_scores[key].get('role_accountability', [])
        if scores:
            substantive = sum(1 for s in scores if s >= 2)
            poll_asr[model][cond] = substantive / len(scores)
        else:
            poll_asr[model][cond] = 0.0

print(f"{'Model':<22} {'ASR(3B)':>10} {'ASR(3C)':>10} {'ΔASR':>10}")
print("-" * 54)
for model in ['gpt-4o', 'claude-3.5-sonnet', 'gemini-1.5-pro']:
    b = poll_asr[model].get('3B', 0)
    c = poll_asr[model].get('3C', 0)
    print(f"{model_display[model]:<22} {b:>9.1%} {c:>9.1%} {c-b:>+9.1%}")

# ============================================================
# PART 2: SELF-REPORT VALIDATION (Dimension 1)
# ============================================================
print("\n" + "=" * 70)
print("PART 2: SELF-REPORT VALIDATION")
print("=" * 70)
print("""
Checking: does the model's self-reported certainty_level match
what it actually wrote in the reformulation?
We use hedge-word detection on the reformulation text itself.
""")

HEDGE_WORDS = {'may', 'might', 'could', 'possibly', 'potentially', 'likely',
               'suggests', 'suggest', 'suggesting', 'appears', 'appear',
               'seems', 'seem', 'probably', 'perhaps', 'hypothesize',
               'indicate', 'indicates', 'possible', 'uncertain', 'unclear',
               'putative', 'presumably', 'speculate', 'propose', 'proposes'}

def has_hedging(text):
    """Detect if reformulation text contains hedge words."""
    if not text:
        return False
    words = set(re.findall(r'\b\w+\b', text.lower()))
    return bool(words & HEDGE_WORDS)

# Check Dimension 1 results
dim1_results = [r for r in valid if r['condition'] in ('1A', '1B', '1C')]

agreement_counts = defaultdict(lambda: {'agree': 0, 'disagree': 0, 'total': 0})

for r in dim1_results:
    pj = r['parsed_json']
    cert = pj.get('certainty_level', '').lower().strip()
    reformulation = pj.get('reformulation', '')
    
    if cert not in ('high', 'medium', 'low'):
        continue
    
    # Self-report says uncertain?
    self_says_uncertain = cert in ('medium', 'low')
    # Actual text contains hedging?
    text_has_hedging = has_hedging(reformulation)
    
    model = r['model']
    agreement_counts[model]['total'] += 1
    
    if self_says_uncertain == text_has_hedging:
        agreement_counts[model]['agree'] += 1
    else:
        agreement_counts[model]['disagree'] += 1

print(f"{'Model':<22} {'Agreement':>12} {'N':>6} {'Validity':>10}")
print("-" * 52)
for model in ['gpt-4o', 'claude-3.5-sonnet', 'gemini-1.5-pro']:
    d = agreement_counts[model]
    rate = d['agree'] / d['total'] if d['total'] > 0 else 0
    print(f"{model_display[model]:<22} {rate:>11.1%} {d['total']:>6} {'OK' if rate > 0.6 else 'LOW':>10}")

# ============================================================
# PART 3: STATISTICAL ANALYSIS
# ============================================================
print("\n" + "=" * 70)
print("PART 3: STATISTICAL ANALYSIS")
print("=" * 70)

def fisher_exact_2x2(a, b, c, d):
    """
    Approximate Fisher's exact test using chi-squared for 2x2 table.
    a b | (condition 1: agree, disagree)
    c d | (condition 2: agree, disagree)
    Returns chi2, p-value approximation, and effect size (phi).
    """
    n = a + b + c + d
    if n == 0:
        return 0, 1.0, 0
    
    # Chi-squared with Yates correction
    expected = ((a+b)*(a+c)/n, (a+b)*(b+d)/n, (c+d)*(a+c)/n, (c+d)*(b+d)/n)
    if any(e < 1 for e in expected):
        return 0, 1.0, 0
    
    chi2 = sum((obs - exp)**2 / exp for obs, exp in zip([a,b,c,d], expected))
    
    # p-value from chi2 (1 df) using approximation
    # P(X > chi2) ≈ erfc(sqrt(chi2/2)) / 2
    p = math.erfc(math.sqrt(chi2/2))
    
    # Phi coefficient (effect size)
    phi = math.sqrt(chi2 / n) if n > 0 else 0
    
    return chi2, p, phi


def cohens_h(p1, p2):
    """Cohen's h for difference between two proportions."""
    h1 = 2 * math.asin(math.sqrt(p1))
    h2 = 2 * math.asin(math.sqrt(p2))
    return h1 - h2


def confidence_interval_proportion(p, n, z=1.96):
    """Wilson score interval for proportion."""
    if n == 0:
        return (0, 0)
    denom = 1 + z**2/n
    center = (p + z**2/(2*n)) / denom
    margin = z * math.sqrt(p*(1-p)/n + z**2/(4*n**2)) / denom
    return (max(0, center - margin), min(1, center + margin))


# --- EAR Statistical Tests ---
print("\n--- Dimension 1: EAR Statistical Tests ---")
print(f"{'Model':<22} {'Contrast':<12} {'Effect(h)':>10} {'Chi2':>8} {'p-value':>10} {'Sig?':>6}")
print("-" * 70)

# Recompute raw counts for stats
def get_ear_counts(condition, model):
    res = [r for r in valid if r['condition'] == condition and r['model'] == model]
    agree = 0
    total = 0
    for r in res:
        pj = r.get('parsed_json', {})
        cert = pj.get('certainty_level', '').lower().strip()
        gt = r['ground_truth_tag']
        if cert not in ('high', 'medium', 'low'):
            continue
        total += 1
        if (gt == 'ASSERTIVE' and cert == 'high') or (gt == 'UNCERTAIN' and cert in ('medium', 'low')):
            agree += 1
    return agree, total - agree, total

for model in ['gpt-4o', 'claude-3.5-sonnet', 'gemini-1.5-pro']:
    # Test 1A vs 1C
    a1, d1, n1 = get_ear_counts('1A', model)
    a2, d2, n2 = get_ear_counts('1C', model)
    p1 = a1/n1 if n1 > 0 else 0
    p2 = a2/n2 if n2 > 0 else 0
    chi2, p, phi = fisher_exact_2x2(a1, d1, a2, d2)
    h = cohens_h(p2, p1)
    sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "ns"
    print(f"{model_display[model]:<22} {'1A→1C':<12} {h:>+10.3f} {chi2:>8.2f} {p:>10.4f} {sig:>6}")
    
    # Test 1C vs 1B
    a1, d1, n1 = get_ear_counts('1C', model)
    a2, d2, n2 = get_ear_counts('1B', model)
    p1 = a1/n1 if n1 > 0 else 0
    p2 = a2/n2 if n2 > 0 else 0
    chi2, p, phi = fisher_exact_2x2(a1, d1, a2, d2)
    h = cohens_h(p2, p1)
    sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "ns"
    print(f"{'':22} {'1C→1B':<12} {h:>+10.3f} {chi2:>8.2f} {p:>10.4f} {sig:>6}")

# --- JEAR Statistical Tests ---
print(f"\n--- Dimension 2: JEAR Statistical Tests ---")
print(f"{'Model':<22} {'Contrast':<12} {'Effect(h)':>10} {'Chi2':>8} {'p-value':>10} {'Sig?':>6}")
print("-" * 70)

def get_jear_counts(condition, model):
    res = [r for r in valid if r['condition'] == condition and r['model'] == model]
    expand = 0
    total = 0
    for r in res:
        pj = r.get('parsed_json', {})
        s = pj.get('jargon_strategy', '').lower().strip()
        if not s:
            continue
        total += 1
        if s == 'expand':
            expand += 1
    return expand, total - expand, total

for model in ['gpt-4o', 'claude-3.5-sonnet', 'gemini-1.5-pro']:
    a1, d1, n1 = get_jear_counts('2A', model)
    a2, d2, n2 = get_jear_counts('2B', model)
    p1 = a1/n1 if n1 > 0 else 0.001  # avoid division issues
    p2 = a2/n2 if n2 > 0 else 0
    chi2, p, phi = fisher_exact_2x2(a1, d1, a2, d2)
    h = cohens_h(min(0.999, p2), max(0.001, p1))
    sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "ns"
    print(f"{model_display[model]:<22} {'2A→2B':<12} {h:>+10.3f} {chi2:>8.2f} {p:>10.4f} {sig:>6}")

# --- Confidence Intervals ---
print(f"\n--- 95% Confidence Intervals (Wilson Score) ---")
print(f"{'Model':<22} {'Metric':<12} {'Point':>8} {'95% CI':>16}")
print("-" * 60)

for model in ['gpt-4o', 'claude-3.5-sonnet', 'gemini-1.5-pro']:
    for cond, label in [('1A', 'EAR-1A'), ('1B', 'EAR-1B'), ('1C', 'EAR-1C')]:
        a, d, n = get_ear_counts(cond, model)
        p = a/n if n > 0 else 0
        lo, hi = confidence_interval_proportion(p, n)
        print(f"{model_display[model]:<22} {label:<12} {p:>7.1%} [{lo:.1%}, {hi:.1%}]")

# ============================================================
# SAVE ALL ANALYSIS
# ============================================================
print("\n" + "=" * 70)
print("SAVING COMPLETE ANALYSIS")
print("=" * 70)

# Save PoLL results
poll_path = f"{OUTPUT_DIR}/poll_judge_results.json"
with open(poll_path, 'w') as f:
    json.dump({
        'metadata': {'total_judged': len(poll_results), 'errors': errors, 'time_seconds': elapsed},
        'results': poll_results,
        'asr_poll': poll_asr
    }, f, indent=2, ensure_ascii=False)
print(f"Saved: {poll_path}")

# Save complete analysis
full_analysis = {
    'poll_asr': poll_asr,
    'self_report_validity': {m: agreement_counts[m] for m in ['gpt-4o', 'claude-3.5-sonnet', 'gemini-1.5-pro']},
    'poll_mean_scores': {},
}
for model in ['gpt-4o', 'claude-3.5-sonnet', 'gemini-1.5-pro']:
    full_analysis['poll_mean_scores'][model] = {}
    for cond in ['3B', '3C']:
        key = (model, cond)
        scores = poll_scores[key]
        n = len(scores.get('epistemic_faithfulness', []))
        if n > 0:
            full_analysis['poll_mean_scores'][model][cond] = {
                'epistemic_faithfulness': sum(scores['epistemic_faithfulness']) / n,
                'audience_adaptation': sum(scores['audience_adaptation']) / n,
                'role_accountability': sum(scores['role_accountability']) / n,
                'n': n
            }

full_path = f"{OUTPUT_DIR}/full_analysis.json"
with open(full_path, 'w') as f:
    json.dump(full_analysis, f, indent=2)
print(f"Saved: {full_path}")

print("\nALL DONE. Ready for README write-up.")
