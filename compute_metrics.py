import os
"""
Metric Computation & Analysis Pipeline
========================================
Computes all three primary metrics from the paper:
  - EAR (Epistemic Agreement Rate) - Dimension 1
  - JEAR (Jargon Expansion Attempt Rate) - Dimension 2
  - ASR (Accountability Signal Rate) - Dimension 3

Plus delta values for contrastive analysis, per-model breakdowns,
cross-label analysis, and qualitative close-reading samples.

Based on: Paper Sections 4.4, 4.6
"""

import json
import csv
from collections import defaultdict

OUTPUT_DIR = "os.path.dirname(os.path.abspath(__file__))"

# ============================================================
# Load results
# ============================================================
print("=" * 70)
print("LOADING EXPERIMENT RESULTS")
print("=" * 70)

with open(f"{OUTPUT_DIR}/experiment_results.json", 'r') as f:
    data = json.load(f)

results = data['results']
metadata = data['metadata']

print(f"Total results: {len(results)}")
print(f"Models: {json.dumps(metadata['models_used'], indent=2)}")
print(f"Parse errors: {metadata['parse_errors']}")
print(f"API errors: {metadata['api_errors']}")

# Filter to successfully parsed results
valid_results = [r for r in results if r.get('parsed_json') and not r.get('parse_error')]
print(f"Valid (parsed) results: {len(valid_results)}")

# Index by condition and model
def get_results(condition=None, model=None, tag=None):
    """Filter results by condition, model, and/or ground truth tag."""
    out = valid_results
    if condition:
        out = [r for r in out if r['condition'] == condition]
    if model:
        out = [r for r in out if r['model'] == model]
    if tag:
        out = [r for r in out if r['ground_truth_tag'] == tag]
    return out

models = ['gpt-4o', 'claude-3.5-sonnet', 'gemini-1.5-pro']
model_display = {'gpt-4o': 'GPT-4o-mini', 'claude-3.5-sonnet': 'Claude 3.5 Haiku', 'gemini-1.5-pro': 'Gemini 2.0 Flash'}

# ============================================================
# DIMENSION 1: Epistemic Agreement Rate (EAR)
# ============================================================
print("\n" + "=" * 70)
print("DIMENSION 1: EPISTEMIC AGREEMENT RATE (EAR)")
print("=" * 70)
print("""
Definition (Paper Section 4.4.2):
  EAR = proportion of outputs where self-reported certainty_level
  is directionally consistent with the ground-truth tag.
  
  ASSERTIVE <-> "high"
  UNCERTAIN <-> "medium" or "low"
  
  Key contrasts:
    ΔEAR(1A→1C) = role-framing effect (the critical finding)
    ΔEAR(1C→1B) = explicit-instruction effect
""")

def compute_ear(results_list):
    """Compute Epistemic Agreement Rate."""
    if not results_list:
        return 0.0, 0, 0
    
    agree = 0
    total = 0
    
    for r in results_list:
        pj = r.get('parsed_json', {})
        if not pj:
            continue
        
        cert = pj.get('certainty_level', '').lower().strip()
        gt = r['ground_truth_tag']
        
        if not cert or cert not in ('high', 'medium', 'low'):
            # Handle edge cases like "high|medium|low" (model returned template)
            continue
        
        total += 1
        
        if gt == 'ASSERTIVE' and cert == 'high':
            agree += 1
        elif gt == 'UNCERTAIN' and cert in ('medium', 'low'):
            agree += 1
    
    rate = agree / total if total > 0 else 0.0
    return rate, agree, total


# Compute EAR for each condition x model
print(f"\n{'Model':<22} {'Cond 1A':>10} {'Cond 1C':>10} {'Cond 1B':>10} {'ΔEAR(A→C)':>12} {'ΔEAR(C→B)':>12}")
print("-" * 78)

ear_data = {}

for model in models:
    ears = {}
    for cond in ['1A', '1C', '1B']:
        res = get_results(condition=cond, model=model)
        rate, agree, total = compute_ear(res)
        ears[cond] = rate
    
    delta_ac = ears['1C'] - ears['1A']
    delta_cb = ears['1B'] - ears['1C']
    
    ear_data[model] = {**ears, 'delta_ac': delta_ac, 'delta_cb': delta_cb}
    
    print(f"{model_display[model]:<22} {ears['1A']:>9.1%} {ears['1C']:>9.1%} {ears['1B']:>9.1%} "
          f"{delta_ac:>+11.1%} {delta_cb:>+11.1%}")

# Cross-label breakdown
print(f"\n--- EAR by Epistemic Label ---")
print(f"{'Model':<22} {'Tag':<12} {'Cond 1A':>10} {'Cond 1C':>10} {'Cond 1B':>10}")
print("-" * 66)

for model in models:
    for tag in ['ASSERTIVE', 'UNCERTAIN']:
        ears = {}
        for cond in ['1A', '1C', '1B']:
            res = get_results(condition=cond, model=model, tag=tag)
            rate, _, _ = compute_ear(res)
            ears[cond] = rate
        print(f"{model_display[model]:<22} {tag:<12} {ears['1A']:>9.1%} {ears['1C']:>9.1%} {ears['1B']:>9.1%}")

# Interpretation
print(f"""
INTERPRETATION:
  ΔEAR(1A→1C) > 0 means role framing ALONE improves epistemic faithfulness
    (evidence of genuine social role awareness, not just instruction-following)
  ΔEAR(1C→1B) > 0 means explicit instruction further helps
  If ΔEAR(1A→1C) ≈ 0 but ΔEAR(1C→1B) > 0: instruction-following without role awareness
""")

# ============================================================
# DIMENSION 2: Jargon Expansion Attempt Rate (JEAR)
# ============================================================
print("=" * 70)
print("DIMENSION 2: JARGON EXPANSION ATTEMPT RATE (JEAR)")
print("=" * 70)
print("""
Definition (Paper Section 4.4.3):
  JEAR = proportion of outputs reporting jargon_strategy = "expand"
  ΔJEAR = JEAR(2B) - JEAR(2A) = effect of DHH audience specification
""")

def compute_jear(results_list):
    """Compute Jargon Expansion Attempt Rate."""
    if not results_list:
        return 0.0, 0, 0
    
    expand = 0
    total = 0
    strategy_counts = defaultdict(int)
    
    for r in results_list:
        pj = r.get('parsed_json', {})
        if not pj:
            continue
        
        strategy = pj.get('jargon_strategy', '').lower().strip()
        if not strategy:
            continue
        
        total += 1
        strategy_counts[strategy] += 1
        
        if strategy == 'expand':
            expand += 1
    
    rate = expand / total if total > 0 else 0.0
    return rate, expand, total, dict(strategy_counts)


print(f"\n{'Model':<22} {'JEAR(2A)':>10} {'JEAR(2B)':>10} {'ΔJEAR':>10}")
print("-" * 54)

jear_data = {}

for model in models:
    jears = {}
    strategies = {}
    for cond in ['2A', '2B']:
        res = get_results(condition=cond, model=model)
        rate, expand, total, strat = compute_jear(res)
        jears[cond] = rate
        strategies[cond] = strat
    
    delta = jears['2B'] - jears['2A']
    jear_data[model] = {**jears, 'delta': delta, 'strategies': strategies}
    
    print(f"{model_display[model]:<22} {jears['2A']:>9.1%} {jears['2B']:>9.1%} {delta:>+9.1%}")

# Strategy distributions
print(f"\n--- Jargon Strategy Distribution ---")
for model in models:
    print(f"\n  {model_display[model]}:")
    for cond in ['2A', '2B']:
        strats = jear_data[model]['strategies'].get(cond, {})
        total = sum(strats.values())
        parts = [f"{k}: {v} ({v/total*100:.0f}%)" for k, v in sorted(strats.items(), key=lambda x: -x[1])]
        print(f"    Condition {cond}: {', '.join(parts)}")

# Expansion reasoning analysis (Condition 2B)
print(f"\n--- Sample Expansion Reasoning (Condition 2B) ---")
for model in models:
    res = get_results(condition='2B', model=model)
    for r in res[:2]:  # 2 samples per model
        pj = r.get('parsed_json', {})
        reasoning = pj.get('expansion_reasoning', 'N/A')
        print(f"\n  [{model_display[model]}] Stimulus {r['stimulus_id']} (STEM: {r['stem_term']})")
        print(f"    Strategy: {pj.get('jargon_strategy', 'N/A')}")
        print(f"    Reasoning: {str(reasoning)[:150]}")

print(f"""
INTERPRETATION:
  ΔJEAR > 0 means models adapt jargon handling when told about DHH audience
  A high JEAR(2B) indicates the model CLAIMS to adapt (necessary but not
  sufficient for genuine audience modeling)
""")

# ============================================================
# DIMENSION 3: Accountability Signal Rate (ASR)
# ============================================================
print("=" * 70)
print("DIMENSION 3: ACCOUNTABILITY SIGNAL RATE (ASR)")
print("=" * 70)
print("""
Definition (Paper Section 4.4.4):
  ASR = proportion of accountability_note fields containing
  substantive dual-party accountability language.
  
  Satisfaction: acknowledges SIMULTANEOUS accountability to both
  the source speaker AND the target audience.
  
  ASR(3A) should be 0% (baseline - no intermediary role)
  ASR(3C) - ASR(3B) = effect of explicit social stakes
""")

# Keywords indicating dual-party accountability
SPEAKER_KEYWORDS = ['speaker', 'original', 'source', 'author', 'researcher',
                     'presenter', 'fidelity', 'faithful', 'preserve', 'maintain',
                     'retain', 'kept', 'intact', 'accuracy']
AUDIENCE_KEYWORDS = ['audience', 'student', 'deaf', 'dhh', 'listener', 'recipient',
                      'accessible', 'accessibility', 'understand', 'comprehension',
                      'clarity', 'cognitive load', 'hearing']

def score_accountability(text):
    """
    Score whether accountability_note reflects dual-party accountability.
    Returns: 1 (triadic - both speaker + audience), 0 (not triadic)
    """
    if not text or not isinstance(text, str):
        return 0
    
    text_lower = text.lower()
    
    has_speaker = any(kw in text_lower for kw in SPEAKER_KEYWORDS)
    has_audience = any(kw in text_lower for kw in AUDIENCE_KEYWORDS)
    
    return 1 if (has_speaker and has_audience) else 0


def compute_asr(results_list):
    """Compute Accountability Signal Rate."""
    if not results_list:
        return 0.0, 0, 0
    
    substantive = 0
    total = 0
    
    for r in results_list:
        pj = r.get('parsed_json', {})
        if not pj:
            continue
        
        note = pj.get('accountability_note', '')
        ep = pj.get('epistemic_preservation', '')
        
        # Combine accountability_note and epistemic_preservation
        combined = f"{note} {ep}".strip()
        
        if not combined:
            total += 1
            continue
        
        total += 1
        substantive += score_accountability(combined)
    
    rate = substantive / total if total > 0 else 0.0
    return rate, substantive, total


print(f"\n{'Model':<22} {'ASR(3A)':>10} {'ASR(3B)':>10} {'ASR(3C)':>10} {'Δ(3B→3C)':>10}")
print("-" * 66)

asr_data = {}

for model in models:
    asrs = {}
    for cond in ['3A', '3B', '3C']:
        res = get_results(condition=cond, model=model)
        rate, sub, total = compute_asr(res)
        asrs[cond] = rate
    
    delta_bc = asrs['3C'] - asrs['3B']
    asr_data[model] = {**asrs, 'delta_bc': delta_bc}
    
    print(f"{model_display[model]:<22} {asrs['3A']:>9.1%} {asrs['3B']:>9.1%} {asrs['3C']:>9.1%} {delta_bc:>+9.1%}")

# Sample accountability notes
print(f"\n--- Sample Accountability Notes ---")
for model in models:
    for cond in ['3B', '3C']:
        res = get_results(condition=cond, model=model)
        if res:
            r = res[0]
            pj = r.get('parsed_json', {})
            note = pj.get('accountability_note', 'N/A')
            print(f"\n  [{model_display[model]}] Condition {cond}, Stimulus {r['stimulus_id']}")
            print(f"    {str(note)[:200]}")

print(f"""
INTERPRETATION:
  ASR(3A) should be ~0% (no intermediary role = no accountability expected)
  ASR(3B) > 0% means role framing triggers accountability language
  ASR(3C) > ASR(3B) means explicit stakes escalate accountability awareness
""")

# ============================================================
# CROSS-MODEL COMPARISON TABLE (Paper Table format)
# ============================================================
print("=" * 70)
print("CROSS-MODEL COMPARISON SUMMARY")
print("=" * 70)

print(f"""
Table: Primary Metrics Across Models and Conditions

{'':=<78}
{'Metric':<20} {'GPT-4o-mini':>18} {'Claude 3.5 Haiku':>18} {'Gemini 2.0 Flash':>18}
{'':-<78}""")

for metric_name, metric_data, key_pairs in [
    ('EAR (1A)', ear_data, [('1A', None)]),
    ('EAR (1C)', ear_data, [('1C', None)]),
    ('EAR (1B)', ear_data, [('1B', None)]),
    ('ΔEAR (A→C)', ear_data, [('delta_ac', None)]),
    ('ΔEAR (C→B)', ear_data, [('delta_cb', None)]),
    ('JEAR (2A)', jear_data, [('2A', None)]),
    ('JEAR (2B)', jear_data, [('2B', None)]),
    ('ΔJEAR', jear_data, [('delta', None)]),
    ('ASR (3A)', asr_data, [('3A', None)]),
    ('ASR (3B)', asr_data, [('3B', None)]),
    ('ASR (3C)', asr_data, [('3C', None)]),
    ('ΔASR (B→C)', asr_data, [('delta_bc', None)]),
]:
    key = key_pairs[0][0]
    values = []
    for model in models:
        v = metric_data[model][key]
        if 'delta' in key.lower() or 'Δ' in metric_name:
            values.append(f"{v:>+.1%}")
        else:
            values.append(f"{v:>.1%}")
    print(f"{'  ' + metric_name:<20} {values[0]:>18} {values[1]:>18} {values[2]:>18}")

print(f"{'':=<78}")

# ============================================================
# QUALITATIVE CLOSE-READING (Paper Section 4.6.3)
# ============================================================
print("\n" + "=" * 70)
print("QUALITATIVE CLOSE-READING ANALYSIS")
print("=" * 70)
print("""
Per the paper (Section 4.6.3), we examine 9 outputs (3 per model)
focusing on:
  1. Epistemic flattening (hedged → assertive)
  2. Spurious hedging (assertive → hedged) 
  3. Role non-compliance (Condition 3C fails to reference stakes)
""")

# Find epistemic flattening examples (UNCERTAIN but model said "high")
print("\n--- Epistemic Flattening Cases (Hedged → Assertive) ---")
for model in models:
    res = get_results(condition='1A', model=model, tag='UNCERTAIN')
    for r in res:
        pj = r.get('parsed_json', {})
        cert = pj.get('certainty_level', '').lower()
        if cert == 'high':
            print(f"\n  [{model_display[model]}] {r['stimulus_id']}")
            print(f"    Original: {r['sentence_text'][:100]}...")
            print(f"    Hedge cues: {r['epistemic_evidence']}")
            print(f"    Model certainty: {cert} (FLATTENED)")
            print(f"    Reformulation: {str(pj.get('reformulation', ''))[:120]}...")
            break

# Find spurious hedging examples (ASSERTIVE but model said "low/medium")
print("\n--- Spurious Hedging Cases (Assertive → Hedged) ---")
for model in models:
    res = get_results(condition='1A', model=model, tag='ASSERTIVE')
    for r in res:
        pj = r.get('parsed_json', {})
        cert = pj.get('certainty_level', '').lower()
        if cert in ('low', 'medium'):
            print(f"\n  [{model_display[model]}] {r['stimulus_id']}")
            print(f"    Original: {r['sentence_text'][:100]}...")
            print(f"    Model certainty: {cert} (SPURIOUS HEDGING)")
            print(f"    Reformulation: {str(pj.get('reformulation', ''))[:120]}...")
            break

# ============================================================
# SAVE ANALYSIS RESULTS
# ============================================================
print("\n" + "=" * 70)
print("SAVING ANALYSIS")
print("=" * 70)

analysis = {
    'ear': {},
    'jear': {},
    'asr': {},
}

for model in models:
    analysis['ear'][model] = ear_data[model]
    analysis['jear'][model] = {k: v for k, v in jear_data[model].items() if k != 'strategies'}
    analysis['asr'][model] = asr_data[model]

analysis['metadata'] = {
    'total_valid_results': len(valid_results),
    'models': metadata['models_used'],
    'model_display_names': model_display,
}

analysis_path = f"{OUTPUT_DIR}/analysis_results.json"
with open(analysis_path, 'w') as f:
    json.dump(analysis, f, indent=2)
print(f"Saved: {analysis_path}")

# Also create a flat summary CSV
summary_rows = []
for model in models:
    row = {
        'model': model,
        'model_actual': metadata['models_used'][model],
        'display_name': model_display[model],
        'EAR_1A': f"{ear_data[model]['1A']:.3f}",
        'EAR_1C': f"{ear_data[model]['1C']:.3f}",
        'EAR_1B': f"{ear_data[model]['1B']:.3f}",
        'dEAR_AC': f"{ear_data[model]['delta_ac']:+.3f}",
        'dEAR_CB': f"{ear_data[model]['delta_cb']:+.3f}",
        'JEAR_2A': f"{jear_data[model]['2A']:.3f}",
        'JEAR_2B': f"{jear_data[model]['2B']:.3f}",
        'dJEAR': f"{jear_data[model]['delta']:+.3f}",
        'ASR_3A': f"{asr_data[model]['3A']:.3f}",
        'ASR_3B': f"{asr_data[model]['3B']:.3f}",
        'ASR_3C': f"{asr_data[model]['3C']:.3f}",
        'dASR_BC': f"{asr_data[model]['delta_bc']:+.3f}",
    }
    summary_rows.append(row)

summary_csv = f"{OUTPUT_DIR}/analysis_summary.csv"
with open(summary_csv, 'w', newline='') as f:
    writer = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
    writer.writeheader()
    writer.writerows(summary_rows)
print(f"Saved: {summary_csv}")

print("\n" + "=" * 70)
print("ANALYSIS COMPLETE")
print("=" * 70)
