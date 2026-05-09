import os
"""
BioScope Corpus Preprocessor for Communicative Accountability Probe
===================================================================
This script parses the BioScope abstracts.xml file and:
1. Extracts all sentences with their plain text
2. Identifies speculation (hedge) cues and negation cues
3. Tags each sentence as ASSERTIVE or UNCERTAIN
4. Filters by the paper's criteria (15-60 tokens, biomedical STEM content)
5. Outputs a structured dataset ready for stimulus selection

Based on: Section 3.1 & 4.3.1 of the ASI Report
"""

import xml.etree.ElementTree as ET
import re
import json
import csv
from collections import Counter

# ============================================================
# STEP 1: Parse the BioScope XML
# ============================================================
print("=" * 70)
print("STEP 1: Parsing BioScope abstracts.xml")
print("=" * 70)

xml_path = "os.path.dirname(os.path.abspath(__file__))/bioscope/abstracts.xml"

# The XML has a DOCTYPE declaration referencing a DTD.
# We'll parse it ignoring the DTD since we just need the elements.
# Read raw XML and strip the DOCTYPE line to avoid DTD resolution issues
with open(xml_path, 'r', encoding='utf-8') as f:
    raw_xml = f.read()

# Remove DOCTYPE declaration to avoid DTD loading
raw_xml = re.sub(r'<!DOCTYPE[^>]+>', '', raw_xml)

root = ET.fromstring(raw_xml)

# ============================================================
# STEP 2: Extract sentences with annotations
# ============================================================
print("\nSTEP 2: Extracting sentences with annotations")
print("-" * 50)

def extract_text_and_cues(sentence_elem):
    """
    Extract plain text and cue information from a sentence element.
    Handles nested <xcope> and <cue> elements.
    """
    # Get full text content (stripping all XML tags)
    raw_text = ET.tostring(sentence_elem, encoding='unicode', method='text')
    # Normalize whitespace
    plain_text = ' '.join(raw_text.split())

    # Find all speculation cues
    speculation_cues = []
    for cue in sentence_elem.iter('cue'):
        if cue.get('type') == 'speculation':
            cue_text = cue.text.strip() if cue.text else ''
            speculation_cues.append(cue_text)

    # Find all negation cues
    negation_cues = []
    for cue in sentence_elem.iter('cue'):
        if cue.get('type') == 'negation':
            cue_text = cue.text.strip() if cue.text else ''
            negation_cues.append(cue_text)

    # Find all xcope elements and extract their full scope text
    scopes = []
    for xcope in sentence_elem.iter('xcope'):
        scope_text = ET.tostring(xcope, encoding='unicode', method='text')
        scope_text = ' '.join(scope_text.split())
        scope_id = xcope.get('id')
        # Determine scope type from the cue inside
        scope_type = None
        for cue in xcope.findall('cue'):
            scope_type = cue.get('type')
            break
        scopes.append({
            'id': scope_id,
            'type': scope_type,
            'text': scope_text
        })

    return plain_text, speculation_cues, negation_cues, scopes


# Iterate through all documents and sentences
all_sentences = []
doc_count = 0

for doc in root.iter('Document'):
    doc_count += 1
    doc_type = doc.get('type', 'unknown')
    doc_id_elem = doc.find('DocID')
    doc_id = doc_id_elem.text.strip() if doc_id_elem is not None and doc_id_elem.text else 'unknown'
    doc_id_type = doc_id_elem.get('type', 'unknown') if doc_id_elem is not None else 'unknown'

    for sent in doc.iter('sentence'):
        sent_id = sent.get('id')
        plain_text, spec_cues, neg_cues, scopes = extract_text_and_cues(sent)

        # Determine epistemic tag
        has_speculation = len(spec_cues) > 0
        has_negation = len(neg_cues) > 0

        # Per CoNLL-2010: a sentence is UNCERTAIN if it contains speculation cues
        epistemic_tag = "UNCERTAIN" if has_speculation else "ASSERTIVE"

        # Token count (simple whitespace split)
        token_count = len(plain_text.split())

        all_sentences.append({
            'sentence_id': sent_id,
            'doc_id': doc_id,
            'doc_id_type': doc_id_type,
            'doc_type': doc_type,
            'plain_text': plain_text,
            'epistemic_tag': epistemic_tag,
            'speculation_cues': spec_cues,
            'negation_cues': neg_cues,
            'has_speculation': has_speculation,
            'has_negation': has_negation,
            'token_count': token_count,
            'scopes': scopes
        })

print(f"Total documents parsed: {doc_count}")
print(f"Total sentences extracted: {len(all_sentences)}")

# ============================================================
# STEP 3: Corpus Statistics
# ============================================================
print("\n" + "=" * 70)
print("STEP 3: Corpus Statistics")
print("=" * 70)

total = len(all_sentences)
uncertain_count = sum(1 for s in all_sentences if s['epistemic_tag'] == 'UNCERTAIN')
assertive_count = sum(1 for s in all_sentences if s['epistemic_tag'] == 'ASSERTIVE')
negation_count = sum(1 for s in all_sentences if s['has_negation'])

print(f"\nEpistemic Tag Distribution:")
print(f"  ASSERTIVE:  {assertive_count:,} ({assertive_count/total*100:.1f}%)")
print(f"  UNCERTAIN:  {uncertain_count:,} ({uncertain_count/total*100:.1f}%)")
print(f"  With negation: {negation_count:,} ({negation_count/total*100:.1f}%)")

# Speculation cue frequency
all_spec_cues = []
for s in all_sentences:
    all_spec_cues.extend(s['speculation_cues'])

print(f"\nTotal speculation cue instances: {len(all_spec_cues)}")
print(f"\nTop 20 speculation cues (hedge words):")
cue_counts = Counter(all_spec_cues)
for cue, count in cue_counts.most_common(20):
    print(f"  '{cue}': {count}")

# Token length distribution
token_counts = [s['token_count'] for s in all_sentences]
print(f"\nToken length statistics:")
print(f"  Min: {min(token_counts)}")
print(f"  Max: {max(token_counts)}")
print(f"  Mean: {sum(token_counts)/len(token_counts):.1f}")
print(f"  Median: {sorted(token_counts)[len(token_counts)//2]}")

# ============================================================
# STEP 4: Apply paper's selection criteria
# ============================================================
print("\n" + "=" * 70)
print("STEP 4: Applying Selection Criteria from Paper Section 4.3.1")
print("=" * 70)
print("""
Criteria:
  1. Biomedical abstract (all our sentences qualify)
  2. Contains at least one STEM-domain term 
  3. Between 15 and 60 tokens in length
  4. Single, self-contained utterance (no unresolved anaphora)
""")

# Filter by token length (15-60 tokens)
filtered = [s for s in all_sentences if 15 <= s['token_count'] <= 60]
print(f"After token length filter (15-60): {len(filtered)} sentences")

# Split into ASSERTIVE and UNCERTAIN pools
assertive_pool = [s for s in filtered if s['epistemic_tag'] == 'ASSERTIVE']
uncertain_pool = [s for s in filtered if s['epistemic_tag'] == 'UNCERTAIN']

print(f"  ASSERTIVE pool: {len(assertive_pool)}")
print(f"  UNCERTAIN pool: {len(uncertain_pool)}")

# Basic anaphora filter: exclude sentences starting with common anaphoric words
anaphoric_starts = ['It ', 'This ', 'These ', 'That ', 'Those ', 'They ', 'Its ', 'Their ',
                     'He ', 'She ', 'Here ', 'Such ']

def has_anaphoric_start(text):
    return any(text.startswith(w) for w in anaphoric_starts)

assertive_clean = [s for s in assertive_pool if not has_anaphoric_start(s['plain_text'])]
uncertain_clean = [s for s in uncertain_pool if not has_anaphoric_start(s['plain_text'])]

print(f"\nAfter basic anaphora filter:")
print(f"  ASSERTIVE pool: {len(assertive_clean)}")
print(f"  UNCERTAIN pool: {len(uncertain_clean)}")

# ============================================================
# STEP 5: Select 40 sentences (20 ASSERTIVE, 20 UNCERTAIN)
# ============================================================
print("\n" + "=" * 70)
print("STEP 5: Selecting 40 Stimulus Sentences")
print("=" * 70)

import random
random.seed(42)  # Reproducibility

# For UNCERTAIN: prefer sentences with clear, unambiguous hedge cues
# (not just "or" which is ambiguous -- only 11.32% speculative in abstracts)
# Priority: may, might, suggest, could, whether, probably, possible, appear, likely, indicate
strong_hedge_cues = {'may', 'might', 'suggest', 'suggesting', 'suggests', 'suggested',
                     'could', 'whether', 'probably', 'possible', 'possibly', 'appear',
                     'appears', 'appeared', 'likely', 'unlikely', 'indicate that',
                     'indicates that', 'indicated that', 'putative', 'presumably',
                     'hypothesize', 'hypothesized', 'propose', 'proposes', 'proposed',
                     'assume', 'assumed', 'imply', 'implies', 'implied', 'potential',
                     'potentially', 'seem', 'seems', 'seemed'}

def has_strong_hedge(sentence):
    """Check if sentence has at least one strong (unambiguous) hedge cue."""
    return any(c.lower() in strong_hedge_cues for c in sentence['speculation_cues'])

# Filter UNCERTAIN pool to prefer strong hedges
uncertain_strong = [s for s in uncertain_clean if has_strong_hedge(s)]
uncertain_weak = [s for s in uncertain_clean if not has_strong_hedge(s)]

print(f"UNCERTAIN with strong hedge cues: {len(uncertain_strong)}")
print(f"UNCERTAIN with weak/ambiguous cues only: {len(uncertain_weak)}")

# Select 20 UNCERTAIN from strong hedge pool
random.shuffle(uncertain_strong)
selected_uncertain = uncertain_strong[:20]

# Select 20 ASSERTIVE
random.shuffle(assertive_clean)
selected_assertive = assertive_clean[:20]

print(f"\nSelected: {len(selected_uncertain)} UNCERTAIN, {len(selected_assertive)} ASSERTIVE")

# ============================================================
# STEP 6: Extract epistemic evidence and identify STEM terms
# ============================================================
print("\n" + "=" * 70)
print("STEP 6: Annotating Selected Sentences")
print("=" * 70)

def identify_stem_term(text):
    """
    Simple heuristic to identify the primary STEM term in a sentence.
    Looks for capitalized multi-word terms, chemical/biological nomenclature,
    and technical noun phrases.
    """
    # Common biomedical/STEM patterns
    patterns = [
        r'\b[A-Z][A-Z0-9]+-[A-Z0-9]+\b',  # e.g., NF-KB, HIV-1, IL-2
        r'\b[A-Z]{2,}[- ]?[0-9]*[a-z]?\b',  # e.g., DNA, RNA, mRNA, TCF1
        r'\b[A-Z][a-z]*[A-Z][a-z]*\b',  # camelCase terms (rare in bio)
        r'\b(?:gene|protein|receptor|kinase|factor|enzyme|antigen|antibod(?:y|ies)|cytokine|inhibitor|promoter|enhancer|transcription|apoptosis|differentiation|phosphorylation|methylation|expression|signaling|pathway|mutation|phenotype|genotype|allele|chromosome|lymphocyte|monocyte|macrophage|leukocyte|neutrophil|platelet|erythrocyte|T cell|B cell|NK cell)\b',
    ]

    # Find all matches
    candidates = []
    for pattern in patterns:
        matches = re.findall(pattern, text)
        candidates.extend(matches)

    # Also look for long technical words (likely STEM terms)
    words = text.split()
    for word in words:
        clean_word = word.strip('.,;:()')
        if len(clean_word) > 8 and clean_word[0].islower() and not clean_word in [
            'previously', 'expression', 'suggesting', 'indicating', 'containing',
            'determined', 'performed', 'resulting', 'important', 'different',
            'presented', 'treatment', 'following', 'including', 'dependent',
            'increased', 'decreased', 'described', 'generated', 'activated',
            'regulated', 'supported', 'confirmed', 'evaluated', 'identified',
            'demonstrate', 'consistent', 'associated', 'additional', 'represents',
            'significant', 'suggesting', 'indicating']:
            candidates.append(clean_word)

    # Return the first strong candidate, or the best guess
    if candidates:
        # Prefer acronyms and specific bio terms
        for c in candidates:
            if re.match(r'^[A-Z]{2,}', c):
                return c
        return candidates[0]
    return None


# Build annotated stimulus records
stimulus_records = []

for idx, sent in enumerate(selected_uncertain + selected_assertive):
    record = {
        'stimulus_id': f"S{idx+1:03d}",
        'source': 'CoNLL-2010',
        'sentence_id': sent['sentence_id'],
        'doc_id': sent['doc_id'],
        'doc_id_type': sent['doc_id_type'],
        'text': sent['plain_text'],
        'epistemic_tag': sent['epistemic_tag'],
        'epistemic_evidence': ', '.join(sent['speculation_cues']) if sent['speculation_cues'] else 'N/A (no hedge cues)',
        'stem_term': identify_stem_term(sent['plain_text']),
        'token_count': sent['token_count'],
        'negation_cues': ', '.join(sent['negation_cues']) if sent['negation_cues'] else 'none'
    }
    stimulus_records.append(record)

# ============================================================
# STEP 7: Display results
# ============================================================
print("\n" + "=" * 70)
print("STEP 7: Selected Stimulus Sentences")
print("=" * 70)

print("\n--- UNCERTAIN (Hedged) Sentences ---\n")
for rec in stimulus_records[:20]:
    print(f"[{rec['stimulus_id']}] {rec['epistemic_tag']} | PMID: {rec['doc_id']}")
    print(f"  Text: {rec['text'][:120]}...")
    print(f"  Hedge cues: {rec['epistemic_evidence']}")
    print(f"  STEM term: {rec['stem_term']}")
    print(f"  Tokens: {rec['token_count']}")
    print()

print("\n--- ASSERTIVE Sentences ---\n")
for rec in stimulus_records[20:]:
    print(f"[{rec['stimulus_id']}] {rec['epistemic_tag']} | PMID: {rec['doc_id']}")
    print(f"  Text: {rec['text'][:120]}...")
    print(f"  STEM term: {rec['stem_term']}")
    print(f"  Tokens: {rec['token_count']}")
    print()

# ============================================================
# STEP 8: Save to CSV and JSON
# ============================================================
print("=" * 70)
print("STEP 8: Saving preprocessed data")
print("=" * 70)

output_dir = "os.path.dirname(os.path.abspath(__file__))"

# Save full extraction as JSON
full_output = {
    'metadata': {
        'source': 'BioScope abstracts.xml',
        'total_sentences_parsed': len(all_sentences),
        'total_documents': doc_count,
        'assertive_count': assertive_count,
        'uncertain_count': uncertain_count,
        'selection_criteria': {
            'token_range': '15-60',
            'anaphora_filtered': True,
            'strong_hedge_preferred': True,
            'random_seed': 42
        }
    },
    'stimulus_sentences': stimulus_records
}

json_path = f"{output_dir}/stimulus_dataset_source1.json"
with open(json_path, 'w', encoding='utf-8') as f:
    json.dump(full_output, f, indent=2, ensure_ascii=False)
print(f"Saved JSON: {json_path}")

# Save as CSV for easy viewing
csv_path = f"{output_dir}/stimulus_dataset_source1.csv"
with open(csv_path, 'w', newline='', encoding='utf-8') as f:
    writer = csv.DictWriter(f, fieldnames=['stimulus_id', 'source', 'sentence_id', 'doc_id',
                                            'doc_id_type', 'text', 'epistemic_tag',
                                            'epistemic_evidence', 'stem_term', 'token_count',
                                            'negation_cues'])
    writer.writeheader()
    writer.writerows(stimulus_records)
print(f"Saved CSV: {csv_path}")

# Save complete parsed corpus for reference
all_corpus_path = f"{output_dir}/bioscope_abstracts_parsed.json"
with open(all_corpus_path, 'w', encoding='utf-8') as f:
    json.dump({
        'total_sentences': len(all_sentences),
        'sentences': all_sentences
    }, f, indent=2, ensure_ascii=False)
print(f"Saved full corpus parse: {all_corpus_path}")

print("\n" + "=" * 70)
print("PREPROCESSING COMPLETE")
print("=" * 70)
print(f"""
Summary:
  - Parsed {doc_count} documents, {len(all_sentences)} total sentences
  - {assertive_count} ASSERTIVE, {uncertain_count} UNCERTAIN
  - Selected 40 stimulus sentences (20 + 20) for Source 1
  - All saved to {output_dir}/
  
Next steps:
  - Source 2: Fetch 20 PubMed OA abstracts (physics/CS/engineering)
  - Combine into final 60-sentence stimulus set
  - Generate 8 prompt conditions x 60 sentences x 3 models = 1440 prompts
""")
