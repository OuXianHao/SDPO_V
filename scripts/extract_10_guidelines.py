import json

path = "/ssd5/xhou/outputs/vllm_guideline_compare_videomme.json"

with open(path, "r", encoding="utf-8") as f:
    data = json.load(f)

picked = []
for video in data:
    vid = video.get("video_id")
    for q in video.get("questions", []):
        g = q.get("exp_guideline_method", {})
        if g.get("status") == "ok":
            picked.append({
                "video_id": vid,
                "question_id": q.get("question_id"),
                "task_type": q.get("task_type"),
                "question": q.get("question"),
                "answer": q.get("answer"),
                "guideline_feedback": g.get("guideline_feedback", ""),
                "guideline_analysis": g.get("guideline_analysis", ""),
                "guideline_response": g.get("response"),
                "guideline_accuracy_score": g.get("accuracy_score"),
                "baseline_response": q.get("exp_correct_rollout_baseline", {}).get("response"),
                "baseline_accuracy_score": q.get("exp_correct_rollout_baseline", {}).get("accuracy_score"),
            })
        if len(picked) >= 10:
            break
    if len(picked) >= 10:
        break

for i, x in enumerate(picked, 1):
    print(f"\n===== CASE {i} =====")
    print("video_id:", x["video_id"])
    print("question_id:", x["question_id"])
    print("task_type:", x["task_type"])
    print("question:", x["question"])
    print("gold:", x["answer"])
    print("baseline:", x["baseline_response"], "acc=", x["baseline_accuracy_score"])
    print("guideline:", x["guideline_response"], "acc=", x["guideline_accuracy_score"])
    print("guideline_feedback:")
    print(x["guideline_feedback"])
    print("guideline_analysis:")
    print(x["guideline_analysis"])