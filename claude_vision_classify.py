"""
Pool detection using Claude Vision (zero-shot classification).

This script sends satellite images to Claude and asks it to identify
whether a swimming pool is present. No training required.

Usage:
    python claude_vision_classify.py --image-folder images/37205
    python claude_vision_classify.py --test  # Run on test set
"""

import os
import sys
import base64
import argparse
import anthropic
import pandas as pd
from pathlib import Path

# Configuration
CONFIDENCE_THRESHOLDS = {
    "high": 0.80,  # >= this = pool or no_pool
    "low": 0.20,   # <= this = no_pool, between = review
}

def encode_image(image_path: str) -> tuple[str, str]:
    """Read and base64 encode an image file."""
    path = Path(image_path)
    suffix = path.suffix.lower()

    media_type_map = {
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".png": "image/png",
        ".gif": "image/gif",
        ".webp": "image/webp",
    }
    media_type = media_type_map.get(suffix, "image/jpeg")

    with open(image_path, "rb") as f:
        image_data = base64.standard_b64encode(f.read()).decode("utf-8")

    return image_data, media_type


def classify_image(client: anthropic.Anthropic, image_path: str) -> dict:
    """
    Send an image to Claude Vision and get pool classification.

    Returns:
        dict with keys: prediction, confidence, reasoning
    """
    image_data, media_type = encode_image(image_path)

    prompt = """Analyze this satellite/aerial image of a residential property.

IMPORTANT: The property being evaluated is outlined with a RED BOUNDARY. Only consider what is INSIDE the red boundary.

Your task: Determine if there is a SWIMMING POOL inside the red boundary.

Rules:
- Look for rectangular or kidney-shaped blue/turquoise features INSIDE the red boundary
- Pools are typically in backyards
- IGNORE pools that are OUTSIDE the red boundary (those belong to neighbors)
- IGNORE driveways, roofs, cars, patios, trampolines, ponds

Respond in this exact format:
CLASSIFICATION: [pool / no_pool / uncertain]
CONFIDENCE: [high / medium / low]
REASONING: [One sentence explaining what you see inside the boundary]"""

    response = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=200,
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": media_type,
                            "data": image_data,
                        },
                    },
                    {
                        "type": "text",
                        "text": prompt,
                    },
                ],
            }
        ],
    )

    # Parse response
    text = response.content[0].text
    result = {
        "raw_response": text,
        "prediction": "uncertain",
        "confidence": "low",
        "reasoning": "",
    }

    for line in text.strip().split("\n"):
        line = line.strip()
        if line.startswith("CLASSIFICATION:"):
            value = line.split(":", 1)[1].strip().lower()
            if "pool" in value and "no" not in value:
                result["prediction"] = "pool"
            elif "no_pool" in value or "no pool" in value:
                result["prediction"] = "no_pool"
            else:
                result["prediction"] = "uncertain"
        elif line.startswith("CONFIDENCE:"):
            result["confidence"] = line.split(":", 1)[1].strip().lower()
        elif line.startswith("REASONING:"):
            result["reasoning"] = line.split(":", 1)[1].strip()

    # Convert confidence to numeric
    confidence_map = {"high": 0.90, "medium": 0.60, "low": 0.30}
    result["confidence_score"] = confidence_map.get(result["confidence"], 0.50)

    return result


def classify_folder(client: anthropic.Anthropic, folder_path: str, limit: int = None) -> pd.DataFrame:
    """Classify all images in a folder."""
    folder = Path(folder_path)
    image_files = list(folder.glob("*.jpg")) + list(folder.glob("*.png"))

    if limit:
        image_files = image_files[:limit]

    results = []
    total = len(image_files)

    for i, image_path in enumerate(image_files, 1):
        print(f"[{i}/{total}] Classifying {image_path.name}...", end=" ", flush=True)

        try:
            result = classify_image(client, str(image_path))
            result["filename"] = image_path.name
            result["filepath"] = str(image_path)
            results.append(result)
            print(f"{result['prediction']} ({result['confidence']})")
        except Exception as e:
            print(f"ERROR: {e}")
            results.append({
                "filename": image_path.name,
                "filepath": str(image_path),
                "prediction": "error",
                "confidence": "low",
                "confidence_score": 0.0,
                "reasoning": str(e),
                "raw_response": "",
            })

    return pd.DataFrame(results)


def run_test_evaluation(client: anthropic.Anthropic, data_dir: str = "data"):
    """Run evaluation on the labeled test set."""
    test_pool_dir = Path(data_dir) / "test" / "pool_in_parcel"
    test_no_pool_dir = Path(data_dir) / "test" / "no_pool"

    print("\n=== Evaluating on TEST SET ===\n")

    # Classify pool images
    print("--- Pool images (ground truth: pool) ---")
    pool_results = classify_folder(client, test_pool_dir)
    pool_results["ground_truth"] = "pool"

    print("\n--- No-pool images (ground truth: no_pool) ---")
    no_pool_results = classify_folder(client, test_no_pool_dir)
    no_pool_results["ground_truth"] = "no_pool"

    # Combine results
    all_results = pd.concat([pool_results, no_pool_results], ignore_index=True)

    # Calculate accuracy
    print("\n=== RESULTS ===\n")

    # Strict accuracy (uncertain counts as wrong)
    correct = (all_results["prediction"] == all_results["ground_truth"]).sum()
    total = len(all_results)
    accuracy = correct / total * 100

    print(f"Total images: {total}")
    print(f"Correct: {correct}")
    print(f"Accuracy: {accuracy:.1f}%")

    # Breakdown
    print("\n--- Confusion Matrix ---")
    print(pd.crosstab(all_results["ground_truth"], all_results["prediction"], margins=True))

    # Save results
    output_file = "claude_vision_test_results.csv"
    all_results.to_csv(output_file, index=False)
    print(f"\nResults saved to {output_file}")

    return all_results


def main():
    parser = argparse.ArgumentParser(description="Pool detection using Claude Vision")
    parser.add_argument("--image-folder", type=str, help="Folder containing images to classify")
    parser.add_argument("--test", action="store_true", help="Run evaluation on test set")
    parser.add_argument("--limit", type=int, help="Limit number of images to process")
    parser.add_argument("--output", type=str, default="claude_vision_predictions.csv", help="Output CSV file")

    args = parser.parse_args()

    # Check for API key
    if not os.getenv("ANTHROPIC_API_KEY"):
        print("ERROR: ANTHROPIC_API_KEY environment variable not set")
        print("Set it with: export ANTHROPIC_API_KEY='your-key-here'")
        sys.exit(1)

    client = anthropic.Anthropic()

    if args.test:
        run_test_evaluation(client)
    elif args.image_folder:
        results = classify_folder(client, args.image_folder, limit=args.limit)
        results.to_csv(args.output, index=False)
        print(f"\nResults saved to {args.output}")

        # Summary
        print("\n--- Summary ---")
        print(results["prediction"].value_counts())
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
