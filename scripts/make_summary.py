#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
from pathlib import Path

INPUT_PATH = Path("/ssd5/xhou/outputs/vllm_guideline_compare_videomme.json")
OUTPUT_PATH = INPUT_PATH.with_name(INPUT_PATH.stem + ".matched.summary.json")


def flatten_questions(data):
    questions = []
    if not isinstance(data, list):
        raise ValueError("Expected top-level JSON to be a list of videos")
    for video in data:
        for q in video.get("questions", []):
            questions.append(q)
    return questions


def is_number(x):
    return isinstance(x, (int, float))


def mean(xs):
    return sum(xs) / len(xs) if xs else 0.0


def safe_get_score(exp, key, default=0.0):
    v = exp.get(key, default)
    return float(v) if is_number(v) else float(default)


def summarize_branch(exps):
    accs = [safe_get_score(e, "accuracy_score", 0.0) for e in exps]
    fmts = [safe_get_score(e, "format_score", 0.0) for e in exps]
    overalls = [safe_get_score(e, "overall_score", 0.0) for e in exps]

    return {
        "num_evaluated": len(exps),
        "accuracy": mean(accs),
        "accuracy_mean": mean(accs),
        "format_rate": mean(fmts),
        "overall_mean": mean(overalls),
    }


def main():
    with open(INPUT_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)

    questions = flatten_questions(data)

    matched = []
    unmatched_stats = {
        "baseline_ok_guideline_not_ok": 0,
        "guideline_ok_baseline_not_ok": 0,
        "both_not_ok": 0,
    }

    for q in questions:
        b = q.get("exp_correct_rollout_baseline", {})
        g = q.get("exp_guideline_method", {})

        b_status = b.get("status", "missing")
        g_status = g.get("status", "missing")

        if b_status == "ok" and g_status == "ok":
            matched.append(q)
        else:
            if b_status == "ok" and g_status != "ok":
                unmatched_stats["baseline_ok_guideline_not_ok"] += 1
            elif g_status == "ok" and b_status != "ok":
                unmatched_stats["guideline_ok_baseline_not_ok"] += 1
            else:
                unmatched_stats["both_not_ok"] += 1

    baseline_exps = [q["exp_correct_rollout_baseline"] for q in matched]
    guideline_exps = [q["exp_guideline_method"] for q in matched]

    baseline_summary = summarize_branch(baseline_exps)
    guideline_summary = summarize_branch(guideline_exps)

    baseline_wins = 0
    guideline_wins = 0
    ties = 0

    accuracy_agree = 0
    accuracy_disagree = 0

    per_question = []

    for q in matched:
        qid = q.get("question_id")
        vid = q.get("video_id", None)

        b = q["exp_correct_rollout_baseline"]
        g = q["exp_guideline_method"]

        b_acc = safe_get_score(b, "accuracy_score", 0.0)
        g_acc = safe_get_score(g, "accuracy_score", 0.0)

        b_fmt = safe_get_score(b, "format_score", 0.0)
        g_fmt = safe_get_score(g, "format_score", 0.0)

        b_overall = safe_get_score(b, "overall_score", 0.0)
        g_overall = safe_get_score(g, "overall_score", 0.0)

        if b_acc > g_acc:
            baseline_wins += 1
            winner = "baseline"
        elif g_acc > b_acc:
            guideline_wins += 1
            winner = "guideline"
        else:
            # accuracy 相同，再用 overall 作为辅助观察
            if b_overall > g_overall:
                winner = "tie_acc_baseline_better_overall"
            elif g_overall > b_overall:
                winner = "tie_acc_guideline_better_overall"
            else:
                winner = "tie"
            ties += 1

        if b_acc == g_acc:
            accuracy_agree += 1
        else:
            accuracy_disagree += 1

        per_question.append({
            "video_id": vid,
            "question_id": qid,
            "baseline_accuracy_score": b_acc,
            "guideline_accuracy_score": g_acc,
            "baseline_format_score": b_fmt,
            "guideline_format_score": g_fmt,
            "baseline_overall_score": b_overall,
            "guideline_overall_score": g_overall,
            "baseline_response": b.get("response"),
            "guideline_response": g.get("response"),
            "winner_by_accuracy": winner,
        })

    summary = {
        "source_json": str(INPUT_PATH),
        "num_videos_found": len(data),
        "num_questions_found": len(questions),
        "num_matched_questions": len(matched),
        "unmatched_breakdown": unmatched_stats,
        "correct_rollout_baseline_matched": baseline_summary,
        "guideline_method_matched": guideline_summary,
        "head_to_head": {
            "baseline_wins": baseline_wins,
            "guideline_wins": guideline_wins,
            "ties": ties,
            "accuracy_agree": accuracy_agree,
            "accuracy_disagree": accuracy_disagree,
        },
        "per_question_examples_preview": per_question[:20],  # 只预览前20条，避免summary太大
    }

    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(f"Saved matched summary to: {OUTPUT_PATH}")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()