"""
Step 6: LLM Proposer — analyzes execution traces and outputs a new HarnessConfig.

Feeds success/failure traces to Claude API, asks it to propose config changes
that fix failures without breaking successes.

Usage:
    python failure_analysis/type_b_memory/meta_harness/propose.py \
        --traces configs/traces_round_0.json \
        --current_config configs/round_0_baseline.json \
        --output configs/round_1.json

Requires ANTHROPIC_API_KEY environment variable.
"""

import argparse
import json
import os
import sys

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
sys.path.insert(0, PROJECT_ROOT)

from failure_analysis.type_b_memory.meta_harness.harness_config import (
    HarnessConfig, load_config, save_config,
)

PROPOSER_PROMPT = """\
You are a harness optimizer for a biomedical retrieval reranking pipeline.

## System Overview
We have a pipeline that reranks document candidates for negation queries (e.g., "drugs NOT indicated for diabetes"). The pipeline has three tunable components controlled by a JSON config:

1. **format_doc**: How documents are shown to the scoring LLM.
   - `field_priority`: Maps negation_pattern → ordered field list. Fields in this list are shown with their content; others are shown as "[has data]" or hidden.
   - `max_items_per_field`: Max entity items per field (default 5).
   - `max_chars`: Max total chars for formatted document.
   - `show_suppressed_as`: How non-priority fields appear ("[has data]" or "" to hide).

2. **prompt**: The prompt template sent to the scoring LLM.
   - `prompt_template`: Main template with {{query}} and {{doc_text}} placeholders.
   - `prompt_suffix`: Extra instruction appended after the template.

3. **alpha_by_type**: Per-answer-type blending weight for LLM vs mFAR scores.
   - `final_score = alpha * llm_norm + (1-alpha) * mfar_norm`
   - Higher alpha = trust LLM more.

## Current Config
{current_config}

## Execution Traces
Below are traces from {n_traces} rerouted queries. Each trace shows what happened when we reranked.

### Success Cases (gold doc rank improved): {n_success}
{success_traces}

### Failure Cases (gold doc rank worsened or unchanged): {n_failure}
{failure_traces}

### Gold Missing (gold doc not in top-100): {n_missing}
These queries cannot be helped by reranking — the correct document wasn't retrieved.

## Available Fields (EXACT names, use spaces NOT underscores)
{available_fields}

CRITICAL: Field names use SPACES, not underscores. For example: "expression present" NOT "expression_present", "associated with" NOT "associated_with", "side effect" NOT "side_effect".

## How field_priority interacts with Stage 2 boost_fields
- When field_priority is empty (`[]`) for a negation_pattern, the pipeline uses per-query boost_fields from Stage 2 (customized per query). This is usually good.
- When field_priority is non-empty, it OVERRIDES Stage 2 boost_fields for ALL queries matching that pattern. Only do this if Stage 2 consistently picks wrong fields.
- Use pattern-specific keys (e.g., "not_indicated", "not_expressed") rather than "default" to avoid overriding all queries.

## Rules
1. Output ONLY a valid JSON object matching the HarnessConfig schema.
2. Include a "rationale" field explaining each change.
3. Be conservative — change ONE aspect at a time (field_priority OR prompt OR alpha, not all at once).
4. Changes should fix failure cases without breaking success cases.
5. Look for systematic patterns grouped by negation_pattern and answer_type.
6. If field_priority is empty for a pattern, the pipeline uses Stage 2 boost_fields. Only override if you see consistent failures from Stage 2's choices.
7. Keep prompt changes minimal and targeted.
8. NEVER use underscores in field names — always use exact names with spaces as listed above.

Output the modified config JSON:"""


def format_trace_summary(trace, max_query_len=100):
    """Format a single trace for the proposer prompt."""
    q = trace["query"][:max_query_len]
    if len(trace["query"]) > max_query_len:
        q += "..."
    return (
        f"  qid={trace['qid']} | {trace['negation_pattern']}|{trace['answer_type']} | "
        f"boost={trace['boost_fields']}\n"
        f"    Q: {q}\n"
        f"    gold_rank: {trace['gold_rank_before']}→{trace['gold_rank_after']} | "
        f"llm_gold={trace['llm_score_gold']} llm_top1={trace['llm_score_top1']} | "
        f"mfar_gold={trace.get('mfar_score_gold', '?'):.4f} "
        f"mfar_top1={trace.get('mfar_score_top1', '?'):.4f}\n"
        f"    gold_fields: {trace['gold_doc_fields_populated']}\n"
    )


def build_proposer_prompt(traces, current_config):
    """Build the full proposer prompt from traces and current config."""
    success = [t for t in traces if t["improved"]]
    failure = [t for t in traces
               if not t["improved"] and t["gold_rank_before"] >= 0]
    missing = [t for t in traces if t["gold_rank_before"] < 0]

    # Sample traces (don't overwhelm the proposer)
    max_examples = 30
    success_sample = success[:max_examples]
    failure_sample = sorted(failure,
                            key=lambda t: t["gold_rank_after"] - t["gold_rank_before"],
                            reverse=True)[:max_examples]

    success_text = "\n".join(format_trace_summary(t) for t in success_sample)
    failure_text = "\n".join(format_trace_summary(t) for t in failure_sample)

    if not success_text.strip():
        success_text = "  (none)"
    if not failure_text.strip():
        failure_text = "  (none)"

    from failure_analysis.utils import RELATION_FIELDS
    fields_text = ", ".join(RELATION_FIELDS)

    return PROPOSER_PROMPT.format(
        current_config=json.dumps(current_config.__dict__ if hasattr(current_config, '__dict__') else current_config, indent=2),
        n_traces=len(traces),
        n_success=len(success),
        success_traces=success_text,
        n_failure=len(failure),
        failure_traces=failure_text,
        n_missing=len(missing),
        available_fields=fields_text,
    )


def call_proposer(prompt, model="claude-sonnet-4-20250514"):
    """Call Claude API to get proposed config."""
    try:
        import anthropic
    except ImportError:
        print("ERROR: pip install anthropic")
        sys.exit(1)

    client = anthropic.Anthropic()
    response = client.messages.create(
        model=model,
        max_tokens=4096,
        messages=[{"role": "user", "content": prompt}],
    )
    return response.content[0].text


def parse_config_from_response(response_text):
    """Extract JSON config from LLM response."""
    # Try to find JSON block
    import re
    # Look for ```json ... ``` blocks
    json_match = re.search(r"```json\s*\n(.*?)\n```", response_text, re.DOTALL)
    if json_match:
        text = json_match.group(1)
    else:
        # Try to find raw JSON object
        json_match = re.search(r"\{.*\}", response_text, re.DOTALL)
        if json_match:
            text = json_match.group(0)
        else:
            raise ValueError("No JSON found in response")

    data = json.loads(text)
    return HarnessConfig(**data)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--traces", required=True, help="Path to traces JSON")
    parser.add_argument("--current_config", required=True, help="Path to current config JSON")
    parser.add_argument("--output", required=True, help="Path to save proposed config")
    parser.add_argument("--model", default="claude-sonnet-4-20250514",
                        help="Claude model for proposer")
    parser.add_argument("--dry_run", action="store_true",
                        help="Print prompt without calling API")
    args = parser.parse_args()

    # Load inputs
    with open(args.traces) as f:
        traces = json.load(f)
    config = load_config(args.current_config)

    print(f"  Traces: {len(traces)}")
    print(f"  Current config: {args.current_config}")

    # Build prompt
    prompt = build_proposer_prompt(traces, config)

    if args.dry_run:
        print("\n" + "="*60)
        print("DRY RUN — Proposer prompt:")
        print("="*60)
        print(prompt)
        return

    # Call proposer
    print(f"  Calling {args.model}...")
    response = call_proposer(prompt, args.model)

    print(f"\n  Raw response:")
    print(response[:500])

    # Parse config
    try:
        new_config = parse_config_from_response(response)
        save_config(new_config, args.output)
        print(f"\n  Proposed config saved → {args.output}")
        print(f"  Rationale: {new_config.rationale}")
    except Exception as e:
        print(f"\n  ERROR parsing response: {e}")
        # Save raw response for debugging
        debug_path = args.output.replace(".json", "_raw_response.txt")
        with open(debug_path, "w") as f:
            f.write(response)
        print(f"  Raw response saved → {debug_path}")
        sys.exit(1)


if __name__ == "__main__":
    main()
