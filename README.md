# Communicative Accountability in Simulated Intermediary AI

CS-UH 3260 | Artificial Social Intelligence | NYU Abu Dhabi | Spring 2026

## What this project is about

When a researcher says "this *might* suggest a link between X and Y," and an interpreter rephrases that as "this *shows* a link between X and Y" - that's a problem. A hypothesis just got turned into a fact.

This project investigates whether large language models preserve epistemic stance (how certain or uncertain a speaker sounds) when acting as intermediaries between a speaker and an audience. We tested this in the context of academic ASL interpreting, where a Deaf student depends entirely on the interpreter's output to understand a lecture.

## What we did

1. Built a dataset of 60 scientific sentences (half hedged/uncertain, half assertive) from the BioScope corpus and PubMed abstracts
2. Ran each sentence through 3 LLMs under 8 prompt conditions (1,440 total calls)
3. Measured whether the models preserved the original speaker's level of certainty

The conditions test:
- Does assigning an "interpreter" role make the model more careful? (1A vs 1C)
- Does explicitly telling it to preserve hedging help? (1B)
- Does it adapt when told about Deaf audience needs? (2A vs 2B)
- Does it show awareness of responsibility to both speaker and audience? (3A/3B/3C)

## Main findings

**Systematic epistemic flattening.** Without explicit instruction, models report "high certainty" for nearly every sentence - even ones containing "might," "may," "suggests," "appears." They default to sounding confident regardless of the source.

**Role framing alone does nothing.** Simply telling the model "you are an academic interpreter" doesn't change behavior. ΔEAR(A→C) ≈ 0 across all models.

**Explicit instruction works.** Telling it to "preserve exactly how certain or uncertain the speaker sounds" improves performance by +17% to +33% (statistically significant for two out of three models).

**Audience adaptation is instruction-dependent.** When informed about DHH cognitive load constraints, all models switch to conceptual expansion 100% of the time. Without that information, they default to fingerspelling.

## Results

| Metric | GPT-4o-mini | Claude 3.5 Haiku | Gemini 2.0 Flash |
|--------|-------------|------------------|------------------|
| EAR baseline (no role) | 50.0% | 76.7% | 51.7% |
| EAR with explicit instruction | 66.7% | 81.7% | 83.3% |
| JEAR without audience info | 0% | 21.7% | 0% |
| JEAR with DHH audience specified | 100% | 100% | 100% |
| Self-report validity | 71.1% | 75.4% | 72.2% |

## What's in this repo

- `final_stimulus_dataset.csv` - 60 annotated sentences with epistemic tags, hedge cues, STEM terms
- `experiment_results.csv` - all 1,440 model responses
- `analysis_summary.csv` - final metrics per model
- `bioscope/` - BioScope corpus (CoNLL-2010 hedge detection source)
- `uncertainty/` - extended uncertainty corpus (biomedical and news domains)

## Pipeline

```bash
python3 preprocess_bioscope.py        # parse corpus
python3 fetch_pubmed_source2.py       # get PubMed sentences
python3 build_final_dataset.py        # combine into 60 stimuli
python3 run_experiment_fast.py        # run 1440 calls (~9 min)
python3 compute_metrics.py            # calculate EAR, JEAR, ASR
python3 poll_and_stats.py             # PoLL judge + significance tests
```

## Next steps

- Rerun with larger, more capable models for publication-quality results
- Scale stimulus set to 150+ sentences
- Incorporate video-based evaluation using How2Sign ASL clips - annotating non-manual markers (eyebrow raise, head tilt, mouthing) that convey epistemic stance in ASL, and comparing text-based output against actual interpreter behavior in video
- Build a multimodal epistemic stance dataset linking English hedge cues to their signed equivalents in recorded interpreter output
- DHH participant evaluation to validate output appropriateness with the Deaf community

## Limitations

Text-only evaluation - no actual ASL video output tested yet. Models used are fast-tier (mini/haiku/flash), not flagships. N=60 gives moderate statistical power but should be larger for publication. No Deaf community members were consulted in the current design phase.
