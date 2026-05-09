import os
"""
Source 2: Fetch PubMed STEM Abstracts (Fixed Queries)
=====================================================
"""

import json
import re
import csv
import urllib.request
import urllib.parse
import xml.etree.ElementTree as ET
import time
import random

random.seed(42)

BASE_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"

# ============================================================
# STEP 1: Search PubMed with simpler, working queries
# ============================================================
print("=" * 70)
print("STEP 1: Searching PubMed for STEM abstracts")
print("=" * 70)

# Broader queries across physics, CS, engineering
queries = {
    'physics': [
        'quantum computing simulation',
        'photonics optical fiber',
        'particle physics detector',
        'astrophysics gravitational waves',
    ],
    'computer_science': [
        'deep learning neural network classification',
        'natural language processing transformer',
        'reinforcement learning algorithm optimization',
        'computer vision object detection',
    ],
    'engineering': [
        'nanotechnology nanoparticles synthesis',
        'materials science polymer composite',
        'biomedical engineering scaffold tissue',
        'semiconductor device fabrication',
    ],
}

all_pmids = {}

for domain, domain_queries in queries.items():
    print(f"\n--- {domain} ---")
    for q in domain_queries:
        params = urllib.parse.urlencode({
            'db': 'pubmed',
            'term': q,
            'retmax': 15,
            'sort': 'relevance',
            'retmode': 'json'
        })
        url = f"{BASE_URL}/esearch.fcgi?{params}"
        
        try:
            req = urllib.request.Request(url, headers={'User-Agent': 'ASL-Project/1.0'})
            with urllib.request.urlopen(req, timeout=15) as resp:
                data = json.loads(resp.read().decode())
                ids = data.get('esearchresult', {}).get('idlist', [])
                print(f"  '{q}': {len(ids)} PMIDs")
                for pid in ids:
                    if pid not in all_pmids:
                        all_pmids[pid] = domain
        except Exception as e:
            print(f"  Error: {e}")
        
        time.sleep(0.4)

print(f"\nTotal unique PMIDs collected: {len(all_pmids)}")

# ============================================================
# STEP 2: Fetch abstracts
# ============================================================
print("\n" + "=" * 70)
print("STEP 2: Fetching abstracts")
print("=" * 70)

pmid_list = list(all_pmids.keys())
all_abstracts = []

batch_size = 50
for i in range(0, len(pmid_list), batch_size):
    batch = pmid_list[i:i+batch_size]
    params = urllib.parse.urlencode({
        'db': 'pubmed',
        'id': ','.join(batch),
        'retmode': 'xml',
        'rettype': 'abstract'
    })
    url = f"{BASE_URL}/efetch.fcgi?{params}"
    
    try:
        req = urllib.request.Request(url, headers={'User-Agent': 'ASL-Project/1.0'})
        with urllib.request.urlopen(req, timeout=30) as resp:
            xml_data = resp.read().decode()
            root = ET.fromstring(xml_data)
            
            for article in root.iter('PubmedArticle'):
                pmid_elem = article.find('.//PMID')
                pmid = pmid_elem.text if pmid_elem is not None else 'unknown'
                
                abstract_elems = article.findall('.//AbstractText')
                if abstract_elems:
                    # Concatenate all abstract text parts
                    parts = []
                    for elem in abstract_elems:
                        text = ET.tostring(elem, encoding='unicode', method='text')
                        if text and text.strip():
                            parts.append(text.strip())
                    
                    abstract_text = ' '.join(parts)
                    
                    if abstract_text and len(abstract_text) > 100:
                        all_abstracts.append({
                            'pmid': pmid,
                            'domain': all_pmids.get(pmid, 'unknown'),
                            'abstract': abstract_text
                        })
    except Exception as e:
        print(f"  Error fetching batch {i//batch_size + 1}: {e}")
    
    time.sleep(0.5)

print(f"Fetched {len(all_abstracts)} abstracts with text")
for domain in ['physics', 'computer_science', 'engineering']:
    count = sum(1 for a in all_abstracts if a['domain'] == domain)
    print(f"  {domain}: {count}")

# ============================================================
# STEP 3: Sentence splitting and epistemic annotation
# ============================================================
print("\n" + "=" * 70)
print("STEP 3: Splitting into sentences and annotating")
print("=" * 70)

def split_sentences(text):
    text = re.sub(r'\b(Dr|Mr|Mrs|Ms|Prof|Fig|Eq|Sr|Jr|et al|i\.e|e\.g|vs)\.',
                  lambda m: m.group().replace('.', '<DOT>'), text)
    sentences = re.split(r'(?<=[.!?])\s+(?=[A-Z])', text)
    sentences = [s.replace('<DOT>', '.') for s in sentences]
    return [s.strip() for s in sentences if len(s.strip()) > 20]

HEDGE_CUES = [
    'may', 'might', 'could', 'would', 'should',
    'suggest', 'suggests', 'suggested', 'suggesting',
    'indicate', 'indicates', 'indicated', 'indicating',
    'appear', 'appears', 'appeared',
    'seem', 'seems', 'seemed',
    'possible', 'possibly', 'probable', 'probably',
    'likely', 'unlikely', 'potential', 'potentially',
    'hypothesize', 'hypothesized', 'propose', 'proposes', 'proposed',
    'putative', 'presumably', 'whether',
    'speculate', 'speculated',
    'imply', 'implies', 'implied',
    'remains unclear', 'not fully understood',
]

def find_hedge_cues(text):
    found = []
    text_lower = text.lower()
    for cue in HEDGE_CUES:
        pattern = r'\b' + re.escape(cue) + r'\b'
        if re.search(pattern, text_lower):
            found.append(cue)
    return found

all_stem_sentences = []

for abstract in all_abstracts:
    sentences = split_sentences(abstract['abstract'])
    for sent in sentences:
        token_count = len(sent.split())
        if 15 <= token_count <= 60:
            hedges = find_hedge_cues(sent)
            tag = "UNCERTAIN" if hedges else "ASSERTIVE"
            all_stem_sentences.append({
                'pmid': abstract['pmid'],
                'domain': abstract['domain'],
                'text': sent,
                'epistemic_tag': tag,
                'hedge_cues': hedges,
                'token_count': token_count
            })

print(f"Total eligible sentences: {len(all_stem_sentences)}")

uncertain_stem = [s for s in all_stem_sentences if s['epistemic_tag'] == 'UNCERTAIN']
assertive_stem = [s for s in all_stem_sentences if s['epistemic_tag'] == 'ASSERTIVE']

print(f"  UNCERTAIN: {len(uncertain_stem)}")
print(f"  ASSERTIVE: {len(assertive_stem)}")

# ============================================================
# STEP 4: Select 20 sentences (10 ASSERTIVE, 10 UNCERTAIN)
# ============================================================
print("\n" + "=" * 70)
print("STEP 4: Selecting 20 Stimulus Sentences")
print("=" * 70)

anaphoric_starts = ['It ', 'This ', 'These ', 'That ', 'Those ', 'They ', 'Its ',
                     'Their ', 'He ', 'She ', 'Here ', 'Such ', 'Our ', 'We ']

def has_anaphoric_start(text):
    return any(text.startswith(w) for w in anaphoric_starts)

uncertain_filtered = [s for s in uncertain_stem if not has_anaphoric_start(s['text'])]
assertive_filtered = [s for s in assertive_stem if not has_anaphoric_start(s['text'])]

print(f"After anaphora filter: {len(uncertain_filtered)} UNCERTAIN, {len(assertive_filtered)} ASSERTIVE")

# Select with domain diversity
def select_diverse(pool, n):
    by_domain = {}
    for s in pool:
        by_domain.setdefault(s['domain'], []).append(s)
    
    selected = []
    domains = list(by_domain.keys())
    
    # Round-robin from each domain
    for d in domains:
        random.shuffle(by_domain[d])
    
    idx = 0
    while len(selected) < n:
        d = domains[idx % len(domains)]
        if by_domain[d]:
            selected.append(by_domain[d].pop(0))
        idx += 1
        if all(len(v) == 0 for v in by_domain.values()):
            break
    
    return selected[:n]

selected_uncertain = select_diverse(uncertain_filtered, 10)
selected_assertive = select_diverse(assertive_filtered, 10)

print(f"\nSelected: {len(selected_uncertain)} UNCERTAIN, {len(selected_assertive)} ASSERTIVE")

# Domain distribution
for tag, pool in [("UNCERTAIN", selected_uncertain), ("ASSERTIVE", selected_assertive)]:
    domains = {}
    for s in pool:
        domains[s['domain']] = domains.get(s['domain'], 0) + 1
    print(f"  {tag}: {domains}")

# ============================================================
# STEP 5: Identify STEM terms and build records
# ============================================================
print("\n" + "=" * 70)
print("STEP 5: Building Source 2 records")
print("=" * 70)

def identify_stem_term(text):
    # Acronyms first
    acronyms = re.findall(r'\b[A-Z][A-Z0-9]{1,}(?:-[A-Z0-9]+)*\b', text)
    if acronyms:
        return acronyms[0]
    
    # Known STEM terms
    stem_patterns = [
        r'\b(?:quantum|neural|algorithm|photon|graphene|nanotube|semiconductor|nanoparticle|'
        r'electrode|catalyst|polymer|spectroscopy|wavelength|resonance|diffraction|entropy|'
        r'topology|bandwidth|convolution|gradient|optimization|regression|classifier|'
        r'embedding|transformer|encoder|decoder|reinforcement|supervised|unsupervised|'
        r'scaffold|biomaterial|transistor|qubit|entanglement|superconducting|'
        r'nanostructure|biosensor|microfluidic|genome|proteomics|metabolomics)\b'
    ]
    
    for pattern in stem_patterns:
        matches = re.findall(pattern, text, re.IGNORECASE)
        if matches:
            return matches[0]
    
    # Fallback: longest word > 8 chars
    words = text.split()
    tech_words = sorted([w.strip('.,;:()') for w in words if len(w.strip('.,;:()')) > 8],
                        key=len, reverse=True)
    return tech_words[0] if tech_words else "N/A"

source2_records = []
for idx, sent in enumerate(selected_uncertain + selected_assertive):
    record = {
        'stimulus_id': f"S{41+idx:03d}",
        'source': 'PubMed_OA',
        'sentence_id': f"PMID_{sent['pmid']}",
        'doc_id': sent['pmid'],
        'doc_id_type': 'PMID',
        'text': sent['text'],
        'epistemic_tag': sent['epistemic_tag'],
        'epistemic_evidence': ', '.join(sent['hedge_cues']) if sent['hedge_cues'] else 'N/A',
        'stem_term': identify_stem_term(sent['text']),
        'token_count': sent['token_count'],
        'negation_cues': 'none',
        'domain': sent['domain']
    }
    source2_records.append(record)

# Display
print("\n--- UNCERTAIN ---\n")
for rec in source2_records[:10]:
    print(f"[{rec['stimulus_id']}] UNCERTAIN | PMID: {rec['doc_id']} | {rec.get('domain')}")
    print(f"  Text: {rec['text'][:130]}...")
    print(f"  Hedge: {rec['epistemic_evidence']} | STEM: {rec['stem_term']}")
    print()

print("--- ASSERTIVE ---\n")
for rec in source2_records[10:]:
    print(f"[{rec['stimulus_id']}] ASSERTIVE | PMID: {rec['doc_id']} | {rec.get('domain')}")
    print(f"  Text: {rec['text'][:130]}...")
    print(f"  STEM: {rec['stem_term']}")
    print()

# ============================================================
# STEP 6: Save
# ============================================================
output_dir = "os.path.dirname(os.path.abspath(__file__))"

json_path = f"{output_dir}/stimulus_dataset_source2.json"
with open(json_path, 'w', encoding='utf-8') as f:
    json.dump({
        'metadata': {
            'source': 'PubMed OA (physics, CS, engineering)',
            'total_abstracts_fetched': len(all_abstracts),
            'total_eligible_sentences': len(all_stem_sentences),
            'random_seed': 42
        },
        'stimulus_sentences': source2_records
    }, f, indent=2, ensure_ascii=False)

csv_path = f"{output_dir}/stimulus_dataset_source2.csv"
with open(csv_path, 'w', newline='', encoding='utf-8') as f:
    fieldnames = ['stimulus_id', 'source', 'sentence_id', 'doc_id', 'doc_id_type',
                  'text', 'epistemic_tag', 'epistemic_evidence', 'stem_term',
                  'token_count', 'negation_cues', 'domain']
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(source2_records)

print(f"\nSaved: {json_path}")
print(f"Saved: {csv_path}")
print("\nDone!")
