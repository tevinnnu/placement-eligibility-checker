#!/usr/bin/env python3
"""
Placement Eligibility Checker
Console-based Python application for screening job announcements
against student eligibility criteria.

Architecture: Console Python + LLM Extraction
Privacy: Student records never sent to LLM - only job post text
"""

import csv
import json
import os
import sys
import re
from dataclasses import dataclass, asdict
from typing import Optional, List, Dict, Any
from datetime import datetime


# =============================================================================
# DATA MODELS
# =============================================================================

@dataclass
class Student:
    """Student record from students.csv"""
    roll_no: str
    name: str
    branch: str
    cgpa: float
    backlogs: int
    batch: int

    @classmethod
    def from_csv_row(cls, row: Dict[str, str]) -> 'Student':
        return cls(
            roll_no=row['roll_no'].strip(),
            name=row['name'].strip(),
            branch=row['branch'].strip().upper(),
            cgpa=float(row['cgpa']),
            backlogs=int(row['backlogs']),
            batch=int(row['batch'])
        )


@dataclass
class JobPost:
    """Job post from jobs.csv"""
    job_id: str
    company: str
    title: str
    description: str

    @classmethod
    def from_csv_row(cls, row: Dict[str, str]) -> 'JobPost':
        return cls(
            job_id=row['job_id'].strip(),
            company=row['company'].strip(),
            title=row['title'].strip(),
            description=row['description'].strip()
        )


@dataclass
class ExtractedCriteria:
    """LLM-extracted eligibility criteria from job post"""
    branches: List[str]
    min_cgpa: Optional[float]
    max_backlogs: Optional[int]
    batch: Optional[int]
    summary: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            'branches': self.branches,
            'min_cgpa': self.min_cgpa,
            'max_backlogs': self.max_backlogs,
            'batch': self.batch,
            'summary': self.summary
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ExtractedCriteria':
        return cls(
            branches=data.get('branches', []),
            min_cgpa=data.get('min_cgpa'),
            max_backlogs=data.get('max_backlogs'),
            batch=data.get('batch'),
            summary=data.get('summary', '')
        )


@dataclass
class JobResult:
    """Processed job with extracted criteria and status"""
    job_id: str
    company: str
    title: str
    description: str
    branches: List[str]
    min_cgpa: Optional[float]
    max_backlogs: Optional[int]
    batch: Optional[int]
    summary: str
    job_status: str  # 'matched' or 'needs_review'
    review_reason: str


@dataclass
class EligibleStudent:
    """Student eligible for a specific job"""
    job_id: str
    company: str
    title: str
    roll_no: str
    name: str
    branch: str
    cgpa: float
    backlogs: int
    batch: int


# =============================================================================
# BRANCH ALIASES (Standard mapping)
# =============================================================================

BRANCH_ALIASES = {
    'AI&DS': 'AIDS',
    'AI AND DS': 'AIDS',
    'ARTIFICIAL INTELLIGENCE AND DATA SCIENCE': 'AIDS',
    'COMPUTER SCIENCE': 'CSE',
    'COMPUTER SCIENCE AND ENGINEERING': 'CSE',
    'INFORMATION TECHNOLOGY': 'IT',
    'ELECTRONICS AND COMMUNICATION': 'ECE',
    'ELECTRONICS AND COMMUNICATION ENGINEERING': 'ECE',
    'ELECTRICAL AND ELECTRONICS': 'EEE',
    'ELECTRICAL AND ELECTRONICS ENGINEERING': 'EEE',
    'MECHANICAL': 'MECH',
    'MECHANICAL ENGINEERING': 'MECH',
    'CIVIL': 'CIVIL',
    'CIVIL ENGINEERING': 'CIVIL',
    'CSE': 'CSE',
    'IT': 'IT',
    'AIDS': 'AIDS',
    'ECE': 'ECE',
    'EEE': 'EEE',
    'MECH': 'MECH',
    'CIVIL': 'CIVIL',
}

VALID_BRANCHES = {'CSE', 'IT', 'AIDS', 'ECE', 'EEE', 'MECH', 'CIVIL', 'ALL'}


def normalize_branch(branch: str) -> str:
    """Normalize branch name to standard code"""
    branch_upper = branch.strip().upper()
    return BRANCH_ALIASES.get(branch_upper, branch_upper)


# =============================================================================
# LLM EXTRACTION (Simulated for offline/testing - replace with actual API call)
# =============================================================================

def extract_criteria_with_llm(job_description: str) -> ExtractedCriteria:
    """
    Extract eligibility criteria from job description using LLM.

    In production, replace this with actual LLM API call (OpenAI, Anthropic, etc.)
    For now, this simulates extraction using regex patterns as a fallback.
    """
    # Try to use LLM if API key is available
    api_key = os.environ.get('OPENAI_API_KEY') or os.environ.get('ANTHROPIC_API_KEY')

    if api_key:
        return _extract_with_llm_api(job_description, api_key)
    else:
        # Fallback: Rule-based extraction for testing/offline use
        return _extract_with_rules(job_description)


def _extract_with_llm_api(description: str, api_key: str) -> ExtractedCriteria:
    """Extract using actual LLM API - placeholder for real implementation"""
    # This would make an actual API call to OpenAI/Anthropic
    # For now, fall back to rule-based
    return _extract_with_rules(description)


def _extract_with_rules(description: str) -> ExtractedCriteria:
    """Rule-based extraction as fallback when LLM unavailable"""
    desc_upper = description.upper()

    # Extract branches
    branches = []
    for alias, standard in BRANCH_ALIASES.items():
        if alias in desc_upper and standard not in branches:
            if standard in VALID_BRANCHES:
                branches.append(standard)

    # Check for "all branches" or "any branch"
    if any(phrase in desc_upper for phrase in ['ANY BRANCH', 'ALL BRANCHES', 'ALL BRANCH', 'ANY STREAM']):
        branches = ['ALL']

    # Extract min_cgpa
    min_cgpa = None
    cgpa_patterns = [
        r'MIN(?:IMUM)?\s*(?:CGPA|C\.G\.P\.A\.)\s*(?:OF|:)?\s*([0-9]\.?[0-9]?)',
        r'CGPA\s*(?:OF|:|>=|>|=|CUTOFF|CUT\s*OFF|MINIMUM|MIN)\s*[:\-]?\s*([0-9]\.?[0-9]?)',
        r'CGPA\s+(?:IS|AT)\s+(?:LEAST\s+)?([0-9]\.?[0-9]?)',
        r'CGPA\s+([0-9]\.?[0-9]?)\s*(?:MINIMUM|MIN|MIN\.|OR\s+HIGHER|OR\s+ABOVE|\+)',
        r'([0-9]\.?[0-9]?)\s*(?:CGPA|C\.G\.P\.A\.)',
        r'(?:MINIMUM|MIN)\s+(?:CGPA|C\.G\.P\.A\.)\s*(?:OF|:)?\s*([0-9]\.?[0-9]?)',
    ]
    for pattern in cgpa_patterns:
        match = re.search(pattern, desc_upper)
        if match:
            try:
                val = float(match.group(1))
                if 0 <= val <= 10:
                    min_cgpa = val
                    break
            except ValueError:
                pass

    # Extract max_backlogs
    max_backlogs = None
    if 'NO BACKLOG' in desc_upper or 'NO ACTIVE BACKLOG' in desc_upper or 'ZERO BACKLOG' in desc_upper:
        max_backlogs = 0
    else:
        backlog_patterns = [
            r'MAX(?:IMUM)?\s*(?:BACKLOG|BACKLOGS)\s*(?:OF|:|<=|<|=)\s*([0-9]+)',
            r'BACKLOG\s*(?:OF|:|<=|<|=)\s*([0-9]+)',
            r'([0-9]+)\s*BACKLOG',
        ]
        for pattern in backlog_patterns:
            match = re.search(pattern, desc_upper)
            if match:
                try:
                    max_backlogs = int(match.group(1))
                    break
                except ValueError:
                    pass

    # Extract batch
    batch = None
    batch_match = re.search(r'\b(20\d{2})\b', description)
    if batch_match:
        year = int(batch_match.group(1))
        if 2020 <= year <= 2035:
            batch = year

    # Generate summary
    summary_parts = []
    if branches:
        if branches == ['ALL']:
            summary_parts.append("All branches")
        else:
            summary_parts.append(f"Branches: {', '.join(branches)}")
    if min_cgpa is not None:
        summary_parts.append(f"CGPA >= {min_cgpa}")
    if max_backlogs is not None:
        summary_parts.append(f"Max backlogs: {max_backlogs}")
    if batch is not None:
        summary_parts.append(f"Batch {batch}")

    summary = "; ".join(summary_parts) if summary_parts else "No specific criteria"
    if len(summary) > 60:  # ~12 words
        summary = summary[:57] + "..."

    return ExtractedCriteria(
        branches=branches,
        min_cgpa=min_cgpa,
        max_backlogs=max_backlogs,
        batch=batch,
        summary=summary
    )


# =============================================================================
# FACT-CHECKING VALIDATION
# =============================================================================

def check_facts(criteria: ExtractedCriteria, job_description: str) -> ExtractedCriteria:
    """
    Verify every LLM-extracted field against raw post text to prevent hallucinations.
    Returns corrected criteria.
    """
    desc_upper = job_description.upper()
    corrected = ExtractedCriteria(
        branches=criteria.branches.copy(),
        min_cgpa=criteria.min_cgpa,
        max_backlogs=criteria.max_backlogs,
        batch=criteria.batch,
        summary=criteria.summary
    )

    # Branch Grounding: Every branch must appear in source post
    if corrected.branches != ['ALL']:
        verified_branches = []
        for branch in corrected.branches:
            # Check if branch or its aliases appear in text
            found = False
            for alias, standard in BRANCH_ALIASES.items():
                if standard == branch and alias in desc_upper:
                    found = True
                    break
            if found:
                verified_branches.append(branch)
        corrected.branches = verified_branches

    # Verify "ALL" branches claim
    if corrected.branches == ['ALL']:
        if not any(phrase in desc_upper for phrase in ['ANY BRANCH', 'ALL BRANCHES', 'ALL BRANCH', 'ANY STREAM']):
            corrected.branches = []

    # Numeric Grounding: min_cgpa exact value must appear in text
    if corrected.min_cgpa is not None:
        cgpa_str = str(corrected.min_cgpa)
        # Check if the exact number appears in context of CGPA (allowing words like cutoff, minimum, min between, and punctuation)
        cgpa_context_pattern = rf'(?:CGPA|C\.G\.P\.A\.)\s*(?:CUTOFF|CUT\s*OFF|MINIMUM|MIN|OF|:|>=|>|=)?\s*[:\-]?\s*{re.escape(cgpa_str)}'
        reverse_pattern = rf'{re.escape(cgpa_str)}\s*(?:CGPA|C\.G\.P\.A\.)'
        if not re.search(cgpa_context_pattern, desc_upper) and not re.search(reverse_pattern, desc_upper):
            corrected.min_cgpa = None

    # Numeric Grounding: batch exact value must appear
    if corrected.batch is not None:
        batch_str = str(corrected.batch)
        if batch_str not in job_description:
            corrected.batch = None

    # Regenerate summary after corrections
    summary_parts = []
    if corrected.branches:
        if corrected.branches == ['ALL']:
            summary_parts.append("All branches")
        else:
            summary_parts.append(f"Branches: {', '.join(corrected.branches)}")
    if corrected.min_cgpa is not None:
        summary_parts.append(f"CGPA >= {corrected.min_cgpa}")
    if corrected.max_backlogs is not None:
        summary_parts.append(f"Max backlogs: {corrected.max_backlogs}")
    if corrected.batch is not None:
        summary_parts.append(f"Batch {corrected.batch}")

    corrected.summary = "; ".join(summary_parts) if summary_parts else "No specific criteria"
    if len(corrected.summary) > 60:
        corrected.summary = corrected.summary[:57] + "..."

    return corrected


def job_status(criteria: ExtractedCriteria) -> tuple[str, str]:
    """
    Determine job status after fact-checking.
    Returns (status, reason)
    """
    if not criteria.branches:
        return 'needs_review', 'No eligible branches found or verified in job post'

    if criteria.min_cgpa is None:
        return 'needs_review', 'No explicit CGPA cut-off found in job post'

    return 'matched', ''


# =============================================================================
# STUDENT ELIGIBILITY EVALUATION
# =============================================================================

def is_eligible(student: Student, criteria: ExtractedCriteria) -> bool:
    """
    Check if student meets all eligibility criteria for a job.
    All applicable conditions must hold.
    """
    # Branch Match
    if criteria.branches != ['ALL']:
        student_branch_norm = normalize_branch(student.branch)
        if student_branch_norm not in criteria.branches:
            return False

    # CGPA Cut-off
    if criteria.min_cgpa is not None:
        if student.cgpa < criteria.min_cgpa:
            return False

    # Backlog Limit
    if criteria.max_backlogs is not None:
        if student.backlogs > criteria.max_backlogs:
            return False

    # Batch Match
    if criteria.batch is not None:
        if student.batch != criteria.batch:
            return False

    return True


# =============================================================================
# CSV PROCESSING
# =============================================================================

def read_students(csv_path: str) -> List[Student]:
    """Read students from CSV file"""
    students = []
    with open(csv_path, 'r', newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            students.append(Student.from_csv_row(row))
    return students


def read_jobs(csv_path: str) -> List[JobPost]:
    """Read job posts from CSV file"""
    jobs = []
    with open(csv_path, 'r', newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            jobs.append(JobPost.from_csv_row(row))
    return jobs


# =============================================================================
# REPORT GENERATION
# =============================================================================

def write_job_summary(jobs: List[JobResult], output_path: str):
    """Write job_summary.csv"""
    fieldnames = ['job_id', 'company', 'title', 'branches', 'min_cgpa',
                  'max_backlogs', 'batch', 'summary', 'job_status', 'review_reason']

    with open(output_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for job in jobs:
            writer.writerow({
                'job_id': job.job_id,
                'company': job.company,
                'title': job.title,
                'branches': '|'.join(job.branches),
                'min_cgpa': job.min_cgpa if job.min_cgpa is not None else '',
                'max_backlogs': job.max_backlogs if job.max_backlogs is not None else '',
                'batch': job.batch if job.batch is not None else '',
                'summary': job.summary,
                'job_status': job.job_status,
                'review_reason': job.review_reason
            })


def write_eligible(eligible_students: List[EligibleStudent], output_path: str):
    """Write eligible.csv"""
    fieldnames = ['job_id', 'company', 'title', 'roll_no', 'name', 'branch',
                  'cgpa', 'backlogs', 'batch']

    with open(output_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for es in eligible_students:
            writer.writerow(asdict(es))


def write_review(jobs: List[JobResult], output_path: str):
    """Write review.csv - jobs needing manual review"""
    review_jobs = [j for j in jobs if j.job_status == 'needs_review']
    fieldnames = ['job_id', 'company', 'title', 'description', 'branches',
                  'min_cgpa', 'max_backlogs', 'batch', 'summary', 'review_reason']

    with open(output_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for job in review_jobs:
            writer.writerow({
                'job_id': job.job_id,
                'company': job.company,
                'title': job.title,
                'description': job.description,
                'branches': '|'.join(job.branches),
                'min_cgpa': job.min_cgpa if job.min_cgpa is not None else '',
                'max_backlogs': job.max_backlogs if job.max_backlogs is not None else '',
                'batch': job.batch if job.batch is not None else '',
                'summary': job.summary,
                'review_reason': job.review_reason
            })


def write_summary(jobs: List[JobResult], eligible_students: List[EligibleStudent],
                  students: List[Student], output_path: str):
    """Write summary.txt"""
    matched_jobs = [j for j in jobs if j.job_status == 'matched']
    review_jobs = [j for j in jobs if j.job_status == 'needs_review']

    # Students with zero matches
    student_match_count = {s.roll_no: 0 for s in students}
    for es in eligible_students:
        student_match_count[es.roll_no] += 1

    zero_match_students = [s for s in students if student_match_count[s.roll_no] == 0]

    with open(output_path, 'w', encoding='utf-8') as f:
        f.write("PLACEMENT ELIGIBILITY CHECKER - SUMMARY REPORT\n")
        f.write("=" * 60 + "\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

        f.write(f"Total Jobs Processed: {len(jobs)}\n")
        f.write(f"  - Matched (ready for distribution): {len(matched_jobs)}\n")
        f.write(f"  - Needs Review: {len(review_jobs)}\n\n")

        f.write(f"Total Students: {len(students)}\n")
        f.write(f"  - Students with >=1 match: {len(students) - len(zero_match_students)}\n")
        f.write(f"  - Students with zero matches: {len(zero_match_students)}\n\n")

        if review_jobs:
            f.write("JOBS REQUIRING MANUAL REVIEW:\n")
            f.write("-" * 40 + "\n")
            for job in review_jobs:
                f.write(f"  {job.job_id} | {job.company} | {job.title}\n")
                f.write(f"    Reason: {job.review_reason}\n")
                f.write(f"    Extracted: {job.summary}\n\n")

        if zero_match_students:
            f.write("STUDENTS WITH ZERO MATCHES:\n")
            f.write("-" * 40 + "\n")
            for s in zero_match_students:
                f.write(f"  {s.roll_no} | {s.name} | {s.branch} | CGPA: {s.cgpa} | Backlogs: {s.backlogs} | Batch: {s.batch}\n")

        f.write("\n" + "=" * 60 + "\n")
        f.write("NOTE: All lists are DRAFTS pending placement officer review.\n")


# =============================================================================
# MAIN PROCESSING PIPELINE
# =============================================================================

def process_jobs(jobs: List[JobPost], students: List[Student]) -> tuple[List[JobResult], List[EligibleStudent]]:
    """Process all jobs through extraction, validation, and eligibility matching"""
    job_results = []
    all_eligible = []

    for job in jobs:
        print(f"Processing: {job.job_id} - {job.company} - {job.title}")

        # Step 1: LLM Extraction
        try:
            extracted = extract_criteria_with_llm(job.description)
        except Exception as e:
            print(f"  LLM extraction failed: {e}")
            extracted = ExtractedCriteria([], None, None, None, "Extraction failed")

        # Step 2: Fact-checking
        verified = check_facts(extracted, job.description)

        # Step 3: Job Status
        status, reason = job_status(verified)

        job_result = JobResult(
            job_id=job.job_id,
            company=job.company,
            title=job.title,
            description=job.description,
            branches=verified.branches,
            min_cgpa=verified.min_cgpa,
            max_backlogs=verified.max_backlogs,
            batch=verified.batch,
            summary=verified.summary,
            job_status=status,
            review_reason=reason
        )
        job_results.append(job_result)

        # Step 4: Student Eligibility (only for matched jobs)
        if status == 'matched':
            for student in students:
                if is_eligible(student, verified):
                    all_eligible.append(EligibleStudent(
                        job_id=job.job_id,
                        company=job.company,
                        title=job.title,
                        roll_no=student.roll_no,
                        name=student.name,
                        branch=student.branch,
                        cgpa=student.cgpa,
                        backlogs=student.backlogs,
                        batch=student.batch
                    ))
            print(f"  Status: MATCHED | Eligible students: {sum(1 for e in all_eligible if e.job_id == job.job_id)}")
        else:
            print(f"  Status: NEEDS REVIEW | Reason: {reason}")

    return job_results, all_eligible


def main():
    """Main entry point"""
    print("=" * 60)
    print("PLACEMENT ELIGIBILITY CHECKER")
    print("=" * 60)

    # Default file paths
    jobs_csv = 'jobs.csv'
    students_csv = 'students.csv'
    output_dir = '.'

    # Allow command line arguments
    if len(sys.argv) > 1:
        jobs_csv = sys.argv[1]
    if len(sys.argv) > 2:
        students_csv = sys.argv[2]
    if len(sys.argv) > 3:
        output_dir = sys.argv[3]

    # Check input files exist
    if not os.path.exists(jobs_csv):
        print(f"Error: Jobs file not found: {jobs_csv}")
        print("Usage: python main.py [jobs.csv] [students.csv] [output_dir]")
        sys.exit(1)

    if not os.path.exists(students_csv):
        print(f"Error: Students file not found: {students_csv}")
        sys.exit(1)

    os.makedirs(output_dir, exist_ok=True)

    # Read data
    print(f"\nReading jobs from: {jobs_csv}")
    jobs = read_jobs(jobs_csv)
    print(f"  Found {len(jobs)} job posts")

    print(f"Reading students from: {students_csv}")
    students = read_students(students_csv)
    print(f"  Found {len(students)} student records")

    # Process
    print("\nProcessing jobs...")
    job_results, eligible_students = process_jobs(jobs, students)

    # Generate reports
    print("\nGenerating reports...")
    write_job_summary(job_results, os.path.join(output_dir, 'job_summary.csv'))
    print("  - job_summary.csv")

    write_eligible(eligible_students, os.path.join(output_dir, 'eligible.csv'))
    print("  - eligible.csv")

    write_review(job_results, os.path.join(output_dir, 'review.csv'))
    print("  - review.csv")

    write_summary(job_results, eligible_students, students, os.path.join(output_dir, 'summary.txt'))
    print("  - summary.txt")

    # Final summary
    matched = sum(1 for j in job_results if j.job_status == 'matched')
    review = sum(1 for j in job_results if j.job_status == 'needs_review')
    zero_matches = len([s for s in students if s.roll_no not in {e.roll_no for e in eligible_students}])

    print("\n" + "=" * 60)
    print("PROCESSING COMPLETE")
    print("=" * 60)
    print(f"Jobs Matched: {matched}")
    print(f"Jobs Needing Review: {review}")
    print(f"Total Eligible Student-Job Pairs: {len(eligible_students)}")
    print(f"Students with Zero Matches: {zero_matches}")
    print(f"\nReports saved to: {output_dir}")
    print("\nNOTE: All lists are DRAFTS pending placement officer review.")


if __name__ == '__main__':
    main()