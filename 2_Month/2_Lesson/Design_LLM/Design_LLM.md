# LLM Training Data Design --- Professional Guide

## Maqsad

Ushbu hujjat **Designing Your LLM Training Data** topshirig'ini
professional data-engineering formatida bajarish uchun yozilgan.

Asosiy bosqichlar:

1.  LLM domain tanlash
2.  3--5 ta data source topish
3.  Data quality baholash
4.  License / usage rights tekshirish
5.  Data provenance tekshirish
6.  Data mixture loyihalash
7.  Final engineering decision

------------------------------------------------------------------------

## 1. LLM Domain

### Example

**My LLM:** Korean--Uzbek Educational LLM

**Main purpose:** Korean va Uzbek tillarida ta'limiy savol-javob,
tarjima, tushuntirish va o'quv materiallarini qayta ishlash.

**Target users:** - Korean tilini o'rganuvchi Uzbek-speaking learners -
Uzbek tilini o'rganuvchi Korean-speaking learners - Teachers / tutors -
Educational content creators

### Domain acceptance criteria

  Criterion           Savol
  ------------------- ----------------------------------------
  Relevance           Data model vazifasiga xizmat qiladimi?
  Language coverage   Kerakli tillar yetarlimi?
  Quality             Matn sifatlimi?
  Safety              Zararli kontent boshqarilganmi?
  Diversity           Topic, genre va style yetarlimi?
  Legal usability     Rights aniqmi?
  Traceability        Original sourcegacha borish mumkinmi?

------------------------------------------------------------------------

## 2. Data Sources

Recommended discovery sources:

  -----------------------------------------------------------------------
  Source                  Role                    Important note
  ----------------------- ----------------------- -----------------------
  FineWeb                 Web-scale corpus        Exact version and
                                                  upstream rights must be
                                                  verified

  Common Crawl            Web crawl / provenance  Raw web content has
                                                  heterogeneous rights
                                                  and quality

  arXiv                   Scientific content      Paper-level
                                                  rights/terms must be
                                                  checked

  The Stack               Source code             Original repository
                                                  licenses apply

  Hugging Face Datasets   Dataset discovery       License and terms are
                                                  dataset-specific
  -----------------------------------------------------------------------

Official links:

-   https://huggingface.co/docs/hub/datasets-overview
-   https://huggingface.co/docs/hub/datasets-cards
-   https://huggingface.co/datasets/HuggingFaceFW/fineweb
-   https://commoncrawl.org/terms-of-use
-   https://huggingface.co/datasets/bigcode/the-stack
-   https://arxiv.org/

------------------------------------------------------------------------

## 3. Data Quality

Minimum dimensions:

-   Relevance
-   Quality
-   Safety
-   Deduplication
-   PII
-   Language fit
-   Provenance

Recommended score: **1--5**

Interpretation:

-   4.0--5.0 → candidate
-   3.0--3.99 → selective use / stronger filtering
-   \<3.0 → reject or investigate

**Quality score is not a legal permission.**

------------------------------------------------------------------------

## 4. License --- CRITICAL

Every dataset must be checked for:

1.  Commercial use
2.  Modification
3.  Redistribution
4.  Attribution
5.  ShareAlike
6.  NonCommercial
7.  NoDerivatives
8.  Additional Terms / Terms of Use
9.  Takedown / opt-out requirements
10. Upstream content rights

Conservative labels:

-   `PASS`
-   `PASS_WITH_CONDITIONS`
-   `REVIEW`
-   `REJECT`

**Rule:** `unknown` never means `yes`.

For The Stack, the dataset documentation explicitly states that source
code comes from repositories with different licenses and that use must
follow the original licenses, including attribution where applicable.

------------------------------------------------------------------------

## 5. Data Provenance

Recommended chain:

``` text
Dataset
  ↓
Dataset Creator
  ↓
Original / Source Data
  ↓
Web Page / File
  ↓
Original Website / Repository
  ↓
License / Terms of Use
  ↓
Training Corpus Record
```

Minimum record fields:

-   `record_id`
-   `dataset_name`
-   `dataset_version`
-   `source_url`
-   `source_domain`
-   `original_creator`
-   `license`
-   `license_url`
-   `retrieved_at`
-   `processing_version`
-   `language`
-   `quality_score`
-   `pii_status`
-   `removal_status`

Provenance is required for auditability, reproducibility, debugging and
takedown/removal workflows.

------------------------------------------------------------------------

## 6. Data Mixture

Example for a Korean/Uzbek educational model:

  Component                                  Weight
  -------------------------------------- ----------
  General web                                   20%
  Education                                     30%
  Korean/Uzbek reference                        25%
  Science                                       10%
  Code                                           5%
  High-quality instructional/synthetic          10%
  **Total**                                **100%**

The exact mixture should be validated experimentally.

A useful principle is:

> **effective contribution = quantity × quality × relevance × legal
> usability × language fit**

Do not simply select the largest dataset.

------------------------------------------------------------------------

## 7. Final Engineering Decision

### USE

High relevance + acceptable quality + clear rights + traceable
provenance.

### NEEDS FURTHER REVIEW

Useful source, but legal, privacy, safety or quality evidence is
incomplete.

### DO NOT USE

Rights conflict, unsafe content, severe quality problems or untraceable
origin.

------------------------------------------------------------------------

# Production Pipeline

``` text
SOURCE DISCOVERY
      ↓
RAW INGESTION
      ↓
LICENSE / RIGHTS GATE
      ↓
PROVENANCE CAPTURE
      ↓
LANGUAGE IDENTIFICATION
      ↓
QUALITY FILTERING
      ↓
PII / SAFETY FILTERING
      ↓
EXACT + FUZZY DEDUPLICATION
      ↓
DOMAIN / LANGUAGE BALANCING
      ↓
TRAIN / VALIDATION / TEST SPLIT
      ↓
CONTAMINATION CHECK
      ↓
MIXTURE WEIGHTING
      ↓
DATASET CARD + AUDIT LOG
      ↓
MODEL TRAINING
      ↓
EVALUATION
      ↓
ITERATE
```

## Governance rules

1.  Never train directly from raw crawls.
2.  Separate raw, cleaned and final datasets.
3.  Preserve source URL and license evidence.
4.  Store processing versions and hashes.
5.  Maintain removal/takedown support.
6.  Keep evaluation data separate from training data.
7.  Audit language and domain distributions.
8.  Version every filtering rule.
9.  Make legal status explicit.
10. Publish a dataset card for released datasets.

------------------------------------------------------------------------

## References

-   Hugging Face Datasets Overview:
    https://huggingface.co/docs/hub/datasets-overview
-   Hugging Face Dataset Cards:
    https://huggingface.co/docs/hub/datasets-cards
-   Hugging Face Licenses:
    https://huggingface.co/docs/hub/repositories-licenses
-   FineWeb: https://huggingface.co/datasets/HuggingFaceFW/fineweb
-   Common Crawl Terms of Use: https://commoncrawl.org/terms-of-use
-   Common Crawl FAQ: https://commoncrawl.org/faq
-   The Stack: https://huggingface.co/datasets/bigcode/the-stack
-   arXiv: https://arxiv.org/

> Legal disclaimer: this is an engineering/data-governance framework,
> not legal advice. For commercial deployment, verify the exact dataset
> version, upstream rights and intended use with qualified legal
> counsel.
