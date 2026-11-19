# Placement Eligibility Checker

A console-based Python application for screening job announcements against student eligibility criteria. Designed for college placement cells to automate the filtering of job postings.

## Features

- **LLM-based Criteria Extraction**: Extracts structured eligibility criteria (branches, CGPA, backlogs, batch) from unstructured job descriptions
- **Deterministic Fact-Checking**: Validates every extracted field against the original job post text to prevent hallucinations
- **Privacy-First Architecture**: Student records (names, roll numbers, grades, backlogs) never leave local Python memory - only job post text is sent to LLM
- **Rule-Based Eligibility Evaluation**: Pure Python numeric comparisons for CGPA, backlogs, branch matching, and batch verification
- **Draft Report Generation**: Produces four review-ready files for placement officer verification

## Architecture

```
jobs.csv + students.csv
        │
        ▼
┌───────────────────┐
│  LLM Extraction   │  ← Only job description text sent
│  (branches, CGPA, │
│  backlogs, batch) │
└─────────┬─────────┘
          ▼
┌───────────────────┐
│  Fact-Checking    │  ← Verify against raw text
│  (grounding)      │
└─────────┬─────────┘
          ▼
┌───────────────────┐
│  Status Assignment│  ← matched / needs_review
└─────────┬─────────┘
          ▼
┌───────────────────┐
│  Eligibility      │  ← Deterministic Python
│  Evaluation       │     (no LLM involved)
└─────────┬─────────┘
          ▼
┌───────────────────┐
│  Report Generation│
│  (4 CSV/TXT files)│
└───────────────────┘
```

## Reports Generated

| File | Description |
|------|-------------|
| `job_summary.csv` | All jobs with extracted criteria and status |
| `eligible.csv` | Student-job pairs where student is eligible |
| `review.csv` | Jobs flagged for manual review with reasons |
| `summary.txt` | Human-readable summary with statistics |

## Installation

```bash
cd placement_eligibility_checker
# No external dependencies required for rule-based extraction
# For LLM API: pip install openai anthropic requests
```

## Usage

```bash
# Basic usage with default files
python main.py

# Custom file paths
python main.py jobs.csv students.csv output_directory/

# With LLM API (set environment variable)
export OPENAI_API_KEY="your-key"
python main.py
```

### Input Files

**jobs.csv** (required columns: `job_id`, `company`, `title`, `description`)
```csv
job_id,company,title,description
JOB001,TechCorp,Software Engineer,"CSE, IT branches. CGPA 7.5+. No backlogs. 2025 batch."
```

**students.csv** (required columns: `roll_no`, `name`, `branch`, `cgpa`, `backlogs`, `batch`)
```csv
roll_no,name,branch,cgpa,backlogs,batch
25CS001,Aarav Sharma,CSE,8.7,0,2025
```

## Eligibility Rules

A student is eligible for a job **only if ALL applicable conditions hold**:

1. **Branch Match**: Student's branch ∈ extracted branches, or branches = `["ALL"]`
2. **CGPA Cut-off**: Student's CGPA ≥ extracted min_cgpa (float comparison)
3. **Backlog Limit**: Student's backlogs ≤ extracted max_backlogs (if specified)
4. **Batch Match**: Student's batch = extracted batch (if specified)

## Fact-Checking Rules

Before eligibility evaluation, every LLM-extracted field is verified against the raw job text:

- **Branches**: Must explicitly appear in post (aliases like AI&DS → AIDS supported). `["ALL"]` only if post says "any branch" or "all branches"
- **CGPA**: Exact numeric value must appear in context of CGPA. Percentage-only requirements → `null`
- **Batch**: Exact 4-digit year must appear in post
- **Unverified fields** are reset to `null`/empty, triggering `needs_review` status

## Edge Cases Handled

| Scenario | Behavior |
|----------|----------|
| Percentage-only CGPA ("70%") | `min_cgpa: null` → needs_review |
| Vague cutoffs ("top 10%") | `min_cgpa: null` → needs_review |
| Prompt injection attempts | Neutralized (student data never sent to LLM) |
| LLM API failure / invalid JSON | One retry → safe fallback to `needs_review` |
| Zero student matches | Listed in `summary.txt` |

## Example Output

```
============================================================
PLACEMENT ELIGIBILITY CHECKER
============================================================

Reading jobs from: jobs.csv
  Found 10 job posts
Reading students from: students.csv
  Found 28 student records

Processing jobs...
Processing: JOB001 - TechCorp Solutions - Software Engineer
  Status: MATCHED | Eligible students: 4
Processing: JOB002 - DataSys Analytics - Data Analyst
  Status: MATCHED | Eligible students: 12
...

Generating reports...
  - job_summary.csv
  - eligible.csv
  - review.csv
  - summary.txt

============================================================
PROCESSING COMPLETE
============================================================
Jobs Matched: 7
Jobs Needing Review: 3
Total Eligible Student-Job Pairs: 45
Students with Zero Matches: 3

Reports saved to: .
NOTE: All lists are DRAFTS pending placement officer review.
```

## Extending for LLM API

Replace the `extract_criteria_with_llm()` function in `main.py` with actual API calls:

```python
def extract_criteria_with_llm(job_description: str) -> ExtractedCriteria:
    import openai
    client = openai.OpenAI()
    
    prompt = f"""Extract eligibility criteria from this job post as JSON:
    {job_description}
    
    Return only these fields:
    - branches: list of CSE/IT/AIDS/ECE/EEE/MECH/CIVIL or ["ALL"]
    - min_cgpa: number 0-10 or null
    - max_backlogs: integer >=0 or null
    - batch: 4-digit year or null
    - summary: <=12 words"""
    
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        response_format={"type": "json_object"}
    )
    return ExtractedCriteria.from_dict(json.loads(response.choices[0].message.content))
```

## License

Academic use - CIT AI-Native Engineering Programme