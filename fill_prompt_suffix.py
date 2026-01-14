#!/usr/bin/env python3
import argparse
import sys


def load_questions(csv_path):
    try:
        import pandas as pd
    except ImportError:
        print("Missing dependency: pandas. Install with `python3 -m pip install pandas openpyxl`.", file=sys.stderr)
        sys.exit(1)

    df = pd.read_csv(csv_path)
    if "Question" not in df.columns:
        print(f"Expected 'Question' column in {csv_path}. Found: {list(df.columns)}", file=sys.stderr)
        sys.exit(1)
    questions = [q for q in df["Question"].tolist() if isinstance(q, str) and q.strip()]
    if not questions:
        print(f"No questions found in {csv_path}.", file=sys.stderr)
        sys.exit(1)
    return questions


def call_lm_studio(base_url, endpoint, model, prompt, temperature, max_tokens, timeout):
    try:
        import requests
    except ImportError:
        print("Missing dependency: requests. Install with `python3 -m pip install requests`.", file=sys.stderr)
        sys.exit(1)

    url = base_url.rstrip("/") + endpoint
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": temperature,
        "stream": False,
    }
    if max_tokens is not None:
        payload["max_tokens"] = max_tokens

    resp = requests.post(url, json=payload, timeout=timeout)
    resp.raise_for_status()
    data = resp.json()

    # OpenAI-compatible: choices[0].message.content (chat) or choices[0].text (completion).
    if isinstance(data, dict) and data.get("choices"):
        choice = data["choices"][0]
        if isinstance(choice, dict):
            msg = choice.get("message")
            if isinstance(msg, dict) and "content" in msg:
                return msg["content"]
            if "text" in choice:
                return choice["text"]

    raise ValueError(f"Unexpected response format from LM Studio: {data}")


def main():
    parser = argparse.ArgumentParser(
        description="Populate the 'Prompt Suffix' column with questions in order per gender type."
    )
    parser.add_argument(
        "--excel",
        default="TalkTuner Gender Testing.xlsx",
        help="Path to the Excel file to update.",
    )
    parser.add_argument(
        "--csv",
        default="TalkTuner gender_questions_for_ai.csv",
        help="Path to the CSV file containing questions.",
    )
    parser.add_argument(
        "--sheet",
        default=0,
        help="Sheet name or index to update (default: first sheet).",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Output file path (defaults to overwrite the input Excel file).",
    )
    parser.add_argument(
        "--no-inplace",
        action="store_true",
        help="Write to a new file instead of overwriting the input Excel file.",
    )
    parser.add_argument(
        "--api-base",
        default="http://127.0.0.1:1234",
        help="LM Studio server base URL.",
    )
    parser.add_argument(
        "--endpoint",
        default="/v1/chat/completions",
        help="LM Studio endpoint path (OpenAI-compatible).",
    )
    parser.add_argument(
        "--model",
        default="llama-2-13b-chat",
        help="Model ID to use in LM Studio.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="Sampling temperature for generation.",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=None,
        help="Optional max tokens for generation.",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=300,
        help="Request timeout in seconds.",
    )
    parser.add_argument(
        "--skip-generate",
        action="store_true",
        help="Only fill suffix/prompt columns; do not call the model.",
    )
    args = parser.parse_args()
    # Default behavior: overwrite input file unless --no-inplace or --output is set.
    inplace = not args.no_inplace and args.output is None
    output_path = args.excel if inplace else (args.output or "TalkTuner Gender Testing - with suffix.xlsx")

    try:
        import pandas as pd
    except ImportError:
        print("Missing dependency: pandas. Install with `python3 -m pip install pandas openpyxl`.", file=sys.stderr)
        sys.exit(1)

    questions = load_questions(args.csv)
    df = pd.read_excel(args.excel, sheet_name=args.sheet)

    if "Response " in df.columns and "Response" not in df.columns:
        df = df.rename(columns={"Response ": "Response"})

    if "Prompt Suffix" not in df.columns:
        print(f"Expected 'Prompt Suffix' column in {args.excel}. Found: {list(df.columns)}", file=sys.stderr)
        sys.exit(1)
    if "Gender Type" not in df.columns or "Gender" not in df.columns:
        print(
            f"Expected 'Gender Type' and 'Gender' columns in {args.excel}. Found: {list(df.columns)}",
            file=sys.stderr,
        )
        sys.exit(1)

    # Ensure text columns can accept strings without dtype warnings.
    df["Prompt Suffix"] = df["Prompt Suffix"].astype("string")
    df["Prompt Prefix"] = df["Prompt Prefix"].astype("string")
    if "Prompt" in df.columns:
        df["Prompt"] = df["Prompt"].astype("string")
    if "Response" in df.columns:
        df["Response"] = df["Response"].astype("string")

    warnings = []
    for gtype in df["Gender Type"].dropna().unique():
        subset = df[df["Gender Type"] == gtype]
        gender_counts = subset["Gender"].value_counts()
        if len(gender_counts) > 1 and gender_counts.nunique() != 1:
            warnings.append(
                f"Gender counts uneven for '{gtype}': {gender_counts.to_dict()}"
            )
        for gender in subset["Gender"].dropna().unique():
            idx = subset[subset["Gender"] == gender].index
            if len(idx) % len(questions) != 0:
                warnings.append(
                    f"Row count for '{gtype}'/{gender} ({len(idx)}) is not divisible by "
                    f"{len(questions)} questions; suffixes will repeat unevenly."
                )
            for i, row_idx in enumerate(idx):
                df.at[row_idx, "Prompt Suffix"] = questions[i % len(questions)]

    if "Prompt Prefix" not in df.columns:
        print(f"Expected 'Prompt Prefix' column in {args.excel}. Found: {list(df.columns)}", file=sys.stderr)
        sys.exit(1)
    if "Prompt" not in df.columns:
        print(f"Expected 'Prompt' column in {args.excel}. Found: {list(df.columns)}", file=sys.stderr)
        sys.exit(1)

    # Concatenate prefix + suffix into Prompt, skipping empty parts.
    def build_prompt(prefix, suffix):
        parts = [p for p in [prefix, suffix] if isinstance(p, str) and p.strip()]
        return " ".join(parts)

    df["Prompt"] = [build_prompt(p, s) for p, s in zip(df["Prompt Prefix"], df["Prompt Suffix"])]

    if not args.skip_generate:
        if "Response" not in df.columns:
            print(f"Expected 'Response' column in {args.excel}. Found: {list(df.columns)}", file=sys.stderr)
            sys.exit(1)
        if "Word Count" not in df.columns:
            print(f"Expected 'Word Count' column in {args.excel}. Found: {list(df.columns)}", file=sys.stderr)
            sys.exit(1)

        for i, prompt in enumerate(df["Prompt"].tolist()):
            existing_response = df.at[df.index[i], "Response"]
            if isinstance(existing_response, str) and existing_response.strip():
                continue
            if not isinstance(prompt, str) or not prompt.strip():
                df.at[df.index[i], "Response"] = ""
                df.at[df.index[i], "Word Count"] = 0
                df.to_excel(output_path, index=False)
                continue
            print(f"Processing row {i + 2}/{len(df) + 1}...", flush=True)
            try:
                response = call_lm_studio(
                    args.api_base,
                    args.endpoint,
                    args.model,
                    prompt,
                    args.temperature,
                    args.max_tokens,
                    args.timeout,
                )
            except Exception as exc:
                print(f"Error on row {i + 2}: {exc}", file=sys.stderr)
                response = ""
            df.at[df.index[i], "Response"] = response
            df.at[df.index[i], "Word Count"] = len(response.split()) if response else 0
            df.to_excel(output_path, index=False)

    df.to_excel(output_path, index=False)

    print(f"Wrote updated file to: {output_path}")
    if warnings:
        print("Warnings:")
        for w in warnings:
            print(f"- {w}")


if __name__ == "__main__":
    main()
