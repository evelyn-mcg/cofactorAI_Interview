
"""
soap_note.py

Standalone script for:
1. Reading a raw clinical transcript.
2. Splitting into token-based chunks with overlap.
3. Cleaning speaker labels and filler words.
4. Summarizing each chunk into SOAP format.
5. Aggregating chunk-level summaries into a final SOAP note.
"""

import re
import argparse
from typing import Pattern, List
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline

# PROMPTS
# subjective - 
PROMPT_S = """
You are a medical note assistant summarizing Subjective information from a patient-provider conversation. Focus on the patient's history, symptoms, and complaints.
Provide a concise summary using complete sentences that effectively captures this subjective information.

Conversation:
"{conversation}" """

#objective
PROMPT_O = """
You are a medical note assistant summarizing Objective data from a patient-provider conversation. 
Focus on measurable information such as physical exam findings, vital signs, and test results.

Provide a concise summary using complete sentences that effectively captures this objective information.
Conversation:
"{conversation}" """

#assessment
PROMPT_A = """
You are a medical note assistant summarizing Assessment data from a patient-provider conversation. Focus on the professional analysis, including diagnoses and differential diagnoses.
Provide a concise summary using complete sentences that effectively captures this assessment information.
Conversation:
"{conversation}" """

#plan
PROMPT_P = """
You are a medical note assistant summarizing the Plan from a patient-provider conversation. Focus on the provider's proposed treatment plan, follow-up steps, and any patient education provided.
Provide a concise summary using complete sentences that effectively captures this plan.
Conversation:
"{conversation}" """

# Aggregation
PROMPT_Ag = """
You are a medical provider writing a summary of {section} information provided by the patient. Please generate a short professional
summary of the following information:

{info} """

# ------- text cleaning ---------------------------
DR_KEYS  = [
    r"Dr\.?\s*Gill", r"Dr\.?\s*Back",
    r"Examiner", r"Interviewer", r"Doctor", r"Provider"
]
PAT_KEYS = [r"Subject", r"Patient"]

DR_PATTERN: Pattern = re.compile(r"(?i)\b(?:%s)\b" % "|".join(DR_KEYS))
PAT_PATTERN: Pattern = re.compile(r"(?i)\b(?:%s)\b" % "|".join(PAT_KEYS))
FILLER_PATTERN: Pattern = re.compile(r"(?i)\b(?:um+|uh+|erm+)\b[,\s]*")
BRACKET_RE = re.compile(
    r"(?i)^\[(Interviewer|Doctor|Provider|Subject|Patient)]\s*:\s*"
)

# Regex for parsing SOAP lines
SOAP_RE = re.compile(r"^(Subjective|Objective|Assessment|Plan)\s*:\s*(.*)$", re.I)
LABELS = ["Subjective", "Objective", "Assessment", "Plan"]

def clean_text(line: str) -> str:
    """Normalize speaker labels and remove filler words in a single line."""
    line = line.strip()
    if m := BRACKET_RE.match(line):
        role = m.group(1).lower()
        repl = "Provider:" if role in {"interviewer", "doctor", "provider"} else "Patient:"
        line = BRACKET_RE.sub(repl + " ", line, count=1)
    line = DR_PATTERN.sub("Provider", line)
    line = PAT_PATTERN.sub("Patient", line)
    line = FILLER_PATTERN.sub("", line)
    line = re.sub(r"^(Patient|Provider)\b(?!\s*:)", r"\1:", line, flags=re.I)
    return re.sub(r"\s{2,}", " ", line).strip()

def clean_chunk(chunk: str) -> str:
    """Apply clean_text to every line in a chunk."""
    lines = chunk.splitlines()
    cleaned = [clean_text(ln) for ln in lines]
    cleaned = [ln for ln in cleaned if ln]
    return "\n".join(cleaned)

def chunk_text(text: str, tokenizer, max_tokens: int = 256, overlap: int = 30) -> List[str]:
    """Split text into overlapping chunks of tokens."""
    ids  = tokenizer.encode(text)
    step = max_tokens - overlap
    return [
        tokenizer.decode(ids[i : i + max_tokens], skip_special_tokens=True)
        for i in range(0, len(ids), step)
    ]

# ---- model application ---------
def summarize_chunk(nlp, prompt_template: str, chunk: str) -> str:
    """Summarize one chunk using the pipeline."""
    prompt = prompt_template.format(conversation=chunk)
    out = nlp(
        prompt,
        max_new_tokens=120,
        do_sample=False,
        truncation=True
    )[0]
    return out["generated_text"].strip()
# -------- multi-chunk processing --------
def aggregate_chunk_summaries(rows: List[dict]) -> dict:
    """
    merge chunk-level summaries into soap-labelled summary
    """
    agg = {lbl: [] for lbl in LABELS}
    for row in rows:
        for lbl in LABELS:
            text = row.get(lbl)
            if text:
                agg[lbl].append(text)
    # Join each list of strings into one block of text
    return {lbl: " ".join(agg[lbl]) for lbl in LABELS}

def refine_sections(
    merged: dict,         # output of aggregate_chunk_summaries (plain merged text)
    nlp,
    prompt_ag: str        # a template that accepts `{section}` and `{info}`
) -> dict:
    """
    summarized merged chunks per soap-label
    """
    refined = {}
    for section in LABELS:
        info = merged[section]
        if not info:
            refined[section] = ""
            continue

        prompt = PROMPT_AG.format(section=section, info=info)
        out = nlp(
            prompt,
            max_new_tokens=100,
            do_sample=False,
            truncation=True,
            max_length=512
        )[0]["generated_text"].strip()
        refined[section] = out
    return refined

def main():
    parser = argparse.ArgumentParser(description="RAG-powered SOAP note generator")
    parser.add_argument("--transcript", required=True, help="Path to raw transcript TXT file")
    parser.add_argument("--model_id", type=str, default="google/flan-t5-small",
                        help="HuggingFace model ID")
    args = parser.parse_args()

    # Load tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(args.model_id)
    model = AutoModelForSeq2SeqLM.from_pretrained(args.model_id)
    nlp = pipeline("text2text-generation", model=model, tokenizer=tokenizer,
                   device=-1, batch_size=4)

    # Read transcript and prepare chunks
    text = open(args.transcript, "r", encoding="utf-8").read()
    raw_chunks = chunk_text(text, tokenizer)
    clean_chunks = [clean_chunk(c) for c in raw_chunks]

    # Summaries per chunk
    chunk_summaries = []
    for chunk in clean_chunks:
        subj = summarize_chunk(nlp, PROMPT_S, chunk)
        obj  = summarize_chunk(nlp, PROMPT_O, chunk)
        ass  = summarize_chunk(nlp, PROMPT_A, chunk)
        plan = summarize_chunk(nlp, PROMPT_P, chunk)
        summaries.append({
            "Subjective": subj,
            "Objective": obj,
            "Assessment": ass,
            "Plan": plan
        })

    # Aggregate into final note
    merged = aggregate_chunk_summaries(chunk_summaries)
    final_note = refine_sections(merged, nlp, PROMPT_Ag)

    # Print final note
    print("\n===== Final SOAP Note =====")
    for sec in LABELS:
        print(f"\n{sec}:")
        print(final_note.get(sec, ""))

if __name__ == "__main__":
    main()
