#!/usr/bin/env python3
import argparse
import json
from collections import Counter, defaultdict


def load_jsonl(path: str):
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
    return records


def main():
    parser = argparse.ArgumentParser(description="Summarize RLSD teacher-reweight dump JSONL.")
    parser.add_argument("path", help="Path to reweight dump jsonl")
    parser.add_argument("--topk", type=int, default=20)
    args = parser.parse_args()

    records = load_jsonl(args.path)
    print(f"num records: {len(records)}")

    n_correct = sum(1 for r in records if r.get("is_correct") is True)
    n_wrong = sum(1 for r in records if r.get("is_correct") is False)
    print(f"correct / wrong: {n_correct} / {n_wrong}")

    before_sum = sum(int(r.get("num_reweighted_tokens_before_answer_mask", 0)) for r in records)
    after_sum = sum(int(r.get("num_reweighted_tokens_after_answer_mask", 0)) for r in records)
    print(f"num_reweighted_tokens_before_answer_mask (sum): {before_sum}")
    print(f"num_reweighted_tokens_after_answer_mask  (sum): {after_sum}")
    print(f"num_reweighted_tokens_after_long_response_skip (sum): {after_sum}")

    skipped_long = [r for r in records if r.get("skip_reweight_due_to_long_response") is True]
    print(f"skip_reweight_due_to_long_response samples: {len(skipped_long)}")
    if skipped_long:
        print("  sample_idx:", [r.get("sample_idx") for r in skipped_long])

    hit_max = [r for r in records if r.get("hit_max_response_len")]
    print(f"hit_max_response_len samples: {len(hit_max)}")
    if hit_max:
        print("  sample_idx:", [r.get("sample_idx") for r in hit_max])

    answer_mask_violation = 0
    for r in records:
        for t in r.get("tokens", []):
            if t.get("answer_mask") is True and abs(float(t.get("w_after_answer_mask", 1.0)) - 1.0) > 1e-8:
                answer_mask_violation += 1
    print(f"answer_mask=true but w_after_answer_mask!=1 anomalies: {answer_mask_violation}")

    token_weight_stats = defaultdict(lambda: {"sum": 0.0, "count": 0})
    for r in records:
        for t in r.get("tokens", []):
            tok = t.get("token", "")
            w = float(t.get("final_token_weight_for_loss", 1.0))
            token_weight_stats[tok]["sum"] += w
            token_weight_stats[tok]["count"] += 1

    avg_weights = []
    for tok, stat in token_weight_stats.items():
        avg = stat["sum"] / max(stat["count"], 1)
        avg_weights.append((tok, avg, stat["count"]))
    avg_weights.sort(key=lambda x: abs(x[1] - 1.0), reverse=True)

    print(f"top changed tokens by final_token_weight_for_loss (top {args.topk}):")
    for tok, avg, cnt in avg_weights[: args.topk]:
        print(f"  token={tok!r:20s} avg_weight={avg:.6f} count={cnt}")

    inactive = Counter(r.get("inactive_reason", "") for r in records)
    print("inactive_reason counts:")
    for k, v in inactive.most_common():
        print(f"  {k or '<active>'}: {v}")


if __name__ == "__main__":
    main()
