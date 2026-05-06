import json
import os


INPUT_PATH = "/data/xhou/datasets/Video-R1-data/Video-R1-260k.json"
OUTPUT_PATH = "/data/xhou/datasets/Video-R1-data/Video-R1-260k_video_mc.json"


def main():
    print(f"Loading data from: {INPUT_PATH}")

    with open(INPUT_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)

    print(f"Total samples: {len(data)}")

    filtered_data = [
        item for item in data
        if item.get("data_type") == "video"
        and item.get("problem_type") == "multiple choice"
    ]

    print(f"Filtered video multiple-choice samples: {len(filtered_data)}")

    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)

    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(filtered_data, f, ensure_ascii=False, indent=2)

    print(f"Saved to: {OUTPUT_PATH}")

    if len(filtered_data) > 0:
        print("\nExample sample:")
        print(json.dumps(filtered_data[0], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()