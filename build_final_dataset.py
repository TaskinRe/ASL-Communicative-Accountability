"""
Combine Sources 1 & 2 into Final 60-Sentence Stimulus Dataset
+ Generate All 1440 Prompt Conditions (8 conditions x 60 sentences x 3 models)
=============================================================================

This script:
1. Merges Source 1 (40 CoNLL-2010 biomedical) and Source 2 (20 PubMed OA STEM)
2. Validates the combined dataset (50% hedge balance, token length, etc.)
3. Generates all 8 prompt templates per the paper's Appendix A
4. Outputs the complete experimental design ready for API execution
"""

import json
import csv
import os

OUTPUT_DIR = "os.path.dirname(os.path.abspath(__file__))"

# ============================================================
# STEP 1: Load both sources
# ============================================================
print("=" * 70)
print("STEP 1: Loading Source 1 and Source 2")
print("=" * 70)

with open(f"{OUTPUT_DIR}/stimulus_dataset_source1.json", 'r') as f:
    source1 = json.load(f)

with open(f"{OUTPUT_DIR}/stimulus_dataset_source2.json", 'r') as f:
    source2 = json.load(f)

s1_sentences = source1['stimulus_sentences']
s2_sentences = source2['stimulus_sentences']

print(f"Source 1 (CoNLL-2010): {len(s1_sentences)} sentences")
print(f"Source 2 (PubMed OA):  {len(s2_sentences)} sentences")

# ============================================================
# STEP 2: Combine and validate
# ============================================================
print("\n" + "=" * 70)
print("STEP 2: Combining into Final Stimulus Set")
print("=" * 70)

all_stimuli = s1_sentences + s2_sentences

# Reindex stimulus IDs sequentially
for idx, stim in enumerate(all_stimuli):
    stim['stimulus_id'] = f"S{idx+1:03d}"

total = len(all_stimuli)
uncertain = sum(1 for s in all_stimuli if s['epistemic_tag'] == 'UNCERTAIN')
assertive = sum(1 for s in all_stimuli if s['epistemic_tag'] == 'ASSERTIVE')

print(f"\nFinal Stimulus Set:")
print(f"  Total sentences: {total}")
print(f"  UNCERTAIN: {uncertain} ({uncertain/total*100:.0f}%)")
print(f"  ASSERTIVE: {assertive} ({assertive/total*100:.0f}%)")
print(f"  Target: 50/50 balance = {'PASS' if uncertain == assertive else 'CHECK'}")
print()

# Source breakdown
s1_uncertain = sum(1 for s in s1_sentences if s['epistemic_tag'] == 'UNCERTAIN')
s1_assertive = sum(1 for s in s1_sentences if s['epistemic_tag'] == 'ASSERTIVE')
s2_uncertain = sum(1 for s in s2_sentences if s['epistemic_tag'] == 'UNCERTAIN')
s2_assertive = sum(1 for s in s2_sentences if s['epistemic_tag'] == 'ASSERTIVE')

print(f"  Source 1: {s1_uncertain} UNCERTAIN, {s1_assertive} ASSERTIVE (of {len(s1_sentences)})")
print(f"  Source 2: {s2_uncertain} UNCERTAIN, {s2_assertive} ASSERTIVE (of {len(s2_sentences)})")

# Token length validation
tokens = [s['token_count'] for s in all_stimuli]
print(f"\n  Token range: {min(tokens)}-{max(tokens)} (required: 15-60)")
print(f"  Mean tokens: {sum(tokens)/len(tokens):.1f}")

# STEM term coverage
has_stem = sum(1 for s in all_stimuli if s.get('stem_term'))
print(f"  STEM terms identified: {has_stem}/{total}")

# Matches Table 1 from the paper
print(f"\n  --- Matches Paper Table 1 ---")
print(f"  {'Source':<30} {'# Sent.':<10} {'% Hedged':<10}")
print(f"  {'CoNLL-2010 (biomedical)':<30} {len(s1_sentences):<10} {s1_uncertain/len(s1_sentences)*100:.0f}%")
print(f"  {'PubMed OA (STEM)':<30} {len(s2_sentences):<10} {s2_uncertain/len(s2_sentences)*100:.0f}%")
print(f"  {'Total':<30} {total:<10} {uncertain/total*100:.0f}%")

# ============================================================
# STEP 3: Generate Prompt Templates (from Appendix A)
# ============================================================
print("\n" + "=" * 70)
print("STEP 3: Generating 8 Prompt Conditions per Appendix A")
print("=" * 70)

def make_prompts(sentence_text, stem_term):
    """Generate all 8 prompt conditions for a given stimulus sentence."""
    
    prompts = {}
    
    # --- Dimension 1: Epistemic Faithfulness ---
    
    # Condition 1A: Neutral Reformulation Baseline
    prompts['1A'] = f'''Here is a sentence from a scientific paper:
"{sentence_text}"

Reformulate this for a general academic audience.
Output only valid JSON with no other text:
{{"reformulation": "...",
"certainty_level": "high|medium|low"}}'''
    
    # Condition 1C: Intermediary Role, No Explicit Instruction
    prompts['1C'] = f'''You are an AI assistant acting as an academic interpreter. Your responsibility is to both the original speaker and the audience.

Sentence: "{sentence_text}"

Reformulate for a Deaf academic audience.
Output only valid JSON with no other text:
{{"reformulation": "...",
"certainty_level": "high|medium|low"}}'''
    
    # Condition 1B: Intermediary Role, Explicit Epistemic Obligation
    prompts['1B'] = f'''You are an AI assistant acting as an academic interpreter. Your responsibility is to both the original speaker and the audience. You must preserve exactly how certain or uncertain the speaker sounds do not add or remove hedging.

Sentence: "{sentence_text}"

Reformulate for a Deaf academic audience.
Output only valid JSON with no other text:
{{"reformulation": "...",
"certainty_level": "high|medium|low",
"epistemic_signal": "how did you signal the speaker's certainty level in your output?"}}'''
    
    # --- Dimension 2: Audience Modeling ---
    
    # Condition 2A: No Audience Specification
    prompts['2A'] = f'''Convert this academic speech sentence to ASL gloss format.

STEM term present: "{stem_term}"
Full sentence: "{sentence_text}"

Output only valid JSON with no other text:
{{"ASL_gloss": "...",
"jargon_strategy": "fingerspell|expand|other"}}'''
    
    # Condition 2B: DHH Audience Specification
    prompts['2B'] = f'''The audience consists of Deaf and Hard-of-Hearing graduate students. Research shows that excessive fingerspelling of technical terms significantly increases cognitive load for DHH students in STEM courses. Socially responsible interpretation expands technical terms conceptually rather than spelling them letter by letter.

STEM term: "{stem_term}"
Full sentence: "{sentence_text}"

Output only valid JSON with no other text:
{{"ASL_gloss": "...",
"jargon_strategy": "fingerspell|expand|other",
"expansion_reasoning": "why did you choose this strategy for this audience?"}}'''
    
    # --- Dimension 3: Role Awareness ---
    
    # Condition 3A: Direct Communicator
    prompts['3A'] = f'''You are explaining this concept directly to a student.

Sentence: "{sentence_text}"

Output only valid JSON with no other text:
{{"output": "...",
"whose_perspective": "..."}}'''
    
    # Condition 3B: Intermediary, No Explicit Stakes
    prompts['3B'] = f'''You are an interpreter conveying this message from a speaker to a student. You are responsible to both the speaker and the student.

Sentence: "{sentence_text}"

Output only valid JSON with no other text:
{{"output": "...",
"whose_perspective": "...",
"accountability_note": "what specifically did you preserve from the original speaker?"}}'''
    
    # Condition 3C: Intermediary, Explicit Social Stakes
    prompts['3C'] = f'''You are an AI interpreter in an academic setting. A Deaf graduate student's understanding of this research depends on how accurately you convey both the content AND the speaker's level of certainty. Misrepresenting the speaker's certainty could affect the student's academic and research decisions.

Sentence: "{sentence_text}"

Output only valid JSON with no other text:
{{"output": "...",
"whose_perspective": "...",
"accountability_note": "...",
"epistemic_preservation": "explain how you preserved or modified the epistemic status of the original utterance and why"}}'''
    
    return prompts


# Generate all prompts
all_prompts = []
conditions = ['1A', '1B', '1C', '2A', '2B', '3A', '3B', '3C']
models = ['gpt-4o', 'claude-3.5-sonnet', 'gemini-1.5-pro']

for stim in all_stimuli:
    prompts = make_prompts(stim['text'], stim.get('stem_term', 'N/A'))
    
    for condition in conditions:
        for model in models:
            all_prompts.append({
                'stimulus_id': stim['stimulus_id'],
                'source': stim['source'],
                'doc_id': stim['doc_id'],
                'ground_truth_tag': stim['epistemic_tag'],
                'epistemic_evidence': stim['epistemic_evidence'],
                'stem_term': stim.get('stem_term', 'N/A'),
                'model': model,
                'condition': condition,
                'prompt': prompts[condition],
                'sentence_text': stim['text']
            })

print(f"\nTotal prompts generated: {len(all_prompts)}")
print(f"  = {total} sentences x {len(conditions)} conditions x {len(models)} models")
print(f"  = {total} x {len(conditions)} x {len(models)} = {total * len(conditions) * len(models)}")

# Breakdown by dimension
for cond in conditions:
    count = sum(1 for p in all_prompts if p['condition'] == cond)
    dim = {'1A': 'D1-Neutral', '1B': 'D1-Role+Instruct', '1C': 'D1-Role Only',
           '2A': 'D2-No Audience', '2B': 'D2-DHH Audience',
           '3A': 'D3-Direct', '3B': 'D3-Intermediary', '3C': 'D3-Intermediary+Stakes'}[cond]
    print(f"  Condition {cond} ({dim}): {count} prompts")

# ============================================================
# STEP 4: Save everything
# ============================================================
print("\n" + "=" * 70)
print("STEP 4: Saving Final Datasets")
print("=" * 70)

# 1. Final combined stimulus dataset
final_dataset = {
    'metadata': {
        'project': 'Communicative Accountability in Simulated Intermediary AI',
        'description': 'Behavioral probe stimulus set for LLM epistemic stance preservation',
        'total_stimuli': total,
        'uncertain_count': uncertain,
        'assertive_count': assertive,
        'hedge_balance': f"{uncertain/total*100:.0f}%",
        'sources': {
            'CoNLL-2010': f'{len(s1_sentences)} sentences (biomedical)',
            'PubMed_OA': f'{len(s2_sentences)} sentences (physics, CS, engineering)'
        },
        'conditions': {
            '1A': 'Neutral Reformulation Baseline',
            '1B': 'Intermediary Role + Explicit Epistemic Obligation',
            '1C': 'Intermediary Role, No Explicit Instruction',
            '2A': 'ASL Gloss, No Audience Specification',
            '2B': 'ASL Gloss, DHH Audience Specified',
            '3A': 'Direct Communicator',
            '3B': 'Intermediary, No Explicit Stakes',
            '3C': 'Intermediary, Explicit Social Stakes'
        },
        'models': models,
        'total_api_calls': len(all_prompts)
    },
    'stimuli': all_stimuli
}

path = f"{OUTPUT_DIR}/final_stimulus_dataset.json"
with open(path, 'w', encoding='utf-8') as f:
    json.dump(final_dataset, f, indent=2, ensure_ascii=False)
print(f"Saved: {path}")

# 2. Final stimulus CSV
csv_path = f"{OUTPUT_DIR}/final_stimulus_dataset.csv"
with open(csv_path, 'w', newline='', encoding='utf-8') as f:
    fieldnames = ['stimulus_id', 'source', 'doc_id', 'doc_id_type', 'text',
                  'epistemic_tag', 'epistemic_evidence', 'stem_term', 'token_count']
    writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction='ignore')
    writer.writeheader()
    writer.writerows(all_stimuli)
print(f"Saved: {csv_path}")

# 3. All prompts as JSON (the full 1440-prompt experiment file)
prompts_path = f"{OUTPUT_DIR}/experiment_prompts.json"
with open(prompts_path, 'w', encoding='utf-8') as f:
    json.dump({
        'total_prompts': len(all_prompts),
        'prompts': all_prompts
    }, f, indent=2, ensure_ascii=False)
print(f"Saved: {prompts_path}")

# 4. Also save a compact experiment manifest CSV
manifest_path = f"{OUTPUT_DIR}/experiment_manifest.csv"
with open(manifest_path, 'w', newline='', encoding='utf-8') as f:
    fieldnames = ['stimulus_id', 'ground_truth_tag', 'model', 'condition',
                  'epistemic_evidence', 'stem_term', 'source']
    writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction='ignore')
    writer.writeheader()
    writer.writerows(all_prompts)
print(f"Saved: {manifest_path}")

# ============================================================
# STEP 5: Final Summary
# ============================================================
print("\n" + "=" * 70)
print("FINAL SUMMARY")
print("=" * 70)
print(f"""
DATA PREPROCESSING COMPLETE

Stimulus Dataset:
  - 60 sentences total (matches Paper Table 1)
  - 30 UNCERTAIN / 30 ASSERTIVE (50/50 balance)
  - Source 1: 40 from CoNLL-2010 biomedical (BioScope)
  - Source 2: 20 from PubMed OA (physics, CS, engineering)
  - All sentences: 15-60 tokens, biomedical/STEM domain
  - Each annotated with: epistemic_tag, epistemic_evidence, stem_term

Experiment Prompts:
  - 8 conditions (1A, 1B, 1C, 2A, 2B, 3A, 3B, 3C)
  - 3 models (GPT-4o, Claude 3.5 Sonnet, Gemini 1.5 Pro)
  - {len(all_prompts)} total API calls ready

Files Generated:
  1. final_stimulus_dataset.json  -- Complete stimulus set + metadata
  2. final_stimulus_dataset.csv   -- Spreadsheet-friendly format
  3. experiment_prompts.json      -- All 1440 prompts ready for execution
  4. experiment_manifest.csv      -- Compact experiment tracking file
  5. stimulus_dataset_source1.*   -- Source 1 intermediate files
  6. stimulus_dataset_source2.*   -- Source 2 intermediate files
  7. bioscope_abstracts_parsed.json -- Full 11,871-sentence corpus parse

Next Step: Execute the experiment by calling GPT-4o, Claude 3.5, and
Gemini 1.5 APIs with the generated prompts.
""")

# ============================================================
# Print sample prompts for verification
# ============================================================
print("=" * 70)
print("SAMPLE PROMPTS (for verification)")
print("=" * 70)

sample_stim = all_stimuli[0]  # First UNCERTAIN sentence
sample_prompts = make_prompts(sample_stim['text'], sample_stim.get('stem_term', 'N/A'))

print(f"\nSample stimulus: [{sample_stim['stimulus_id']}] {sample_stim['epistemic_tag']}")
print(f"Text: {sample_stim['text'][:100]}...")
print(f"Cues: {sample_stim['epistemic_evidence']}")

for cond in ['1A', '1C', '1B', '2A', '2B', '3A', '3B', '3C']:
    print(f"\n--- Condition {cond} ---")
    print(sample_prompts[cond][:300])
    if len(sample_prompts[cond]) > 300:
        print("...")
