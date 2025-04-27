import json
import numpy as np
from collections import Counter
import os
import argparse


def convert_rationales_to_spans(rationales, tokens):
    """
    Convert binary rationales to spans of consecutive 1s.

    Args:
        rationales: List of binary rationales (list of lists of 0s and 1s)
        tokens: List of tokens

    Returns:
        List of spans [start_idx, end_idx]
    """
    if not rationales or not tokens:
        return []

    # Make sure all rationales have the same length as tokens
    valid_rationales = []
    for r in rationales:
        if len(r) == len(tokens):
            valid_rationales.append(r)
        else:
            # Try to truncate or pad if there's a small difference
            if len(r) < len(tokens):
                r = r + [0] * (len(tokens) - len(r))
                valid_rationales.append(r)
            elif len(r) > len(tokens):
                r = r[: len(tokens)]
                valid_rationales.append(r)

    if not valid_rationales:
        return []

    # Take majority vote for each token
    aggregated = []
    for i in range(len(tokens)):
        votes = [r[i] for r in valid_rationales if i < len(r)]
        if votes and sum(votes) > len(votes) / 2:  # Majority voted 1
            aggregated.append(1)
        else:
            aggregated.append(0)

    # Convert to spans
    spans = []
    in_span = False
    start_idx = -1

    for i, val in enumerate(aggregated):
        if val == 1 and not in_span:
            in_span = True
            start_idx = i
        elif val == 0 and in_span:
            in_span = False
            spans.append([start_idx, i - 1])

    # Handle the case where the span extends to the end
    if in_span:
        spans.append([start_idx, len(aggregated) - 1])

    return spans


def determine_label(annotators):
    """
    Determine the majority label from annotators.

    Args:
        annotators: List of annotator objects with 'label' field

    Returns:
        0 for normal, 1 for offensive, 2 for hatespeech
    """
    label_map = {"normal": 0, "offensive": 1, "hatespeech": 2}
    labels = [label_map[ann["label"]] for ann in annotators]
    counter = Counter(labels)

    # If there's a tie, prioritize the more severe label
    if (
        len(counter) > 1
        and counter.most_common(1)[0][1] == counter.most_common(2)[1][1]
    ):
        for priority_label in [2, 1, 0]:  # Priority: hatespeech > offensive > normal
            if priority_label in labels:
                return priority_label

    # Otherwise, return the most common label
    return counter.most_common(1)[0][0]


def preprocess_hatexplain_file(input_file_path):
    """
    Preprocess HateXplain dataset from a file to JSONL format.

    Args:
        input_file_path: Path to the JSON file containing HateXplain data

    Returns:
        Dictionary with train, valid, and test JSONL data
    """
    # Load the JSON data
    with open(input_file_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Lists to store processed entries
    processed_entries = []

    # Process each post
    for post_id, post_data in data.items():
        tokens = post_data.get("post_tokens", [])

        # Skip if tokens is empty
        if not tokens:
            continue

        # Get the majority label
        annotators = post_data.get("annotators", [])
        if not annotators:
            continue

        label = determine_label(annotators)

        # Convert rationales to spans
        rationales = post_data.get("rationales", [])
        spans = convert_rationales_to_spans(rationales, tokens)

        # Create the entry based on whether rationales exist
        if spans:
            entry = [tokens, label, spans]
        else:
            entry = [tokens, label]

        processed_entries.append(entry)

    # Split the data into train/valid/test
    np.random.seed(42)  # For reproducibility
    np.random.shuffle(processed_entries)

    n = len(processed_entries)
    train_size = int(n * 0.7)
    valid_size = int(n * 0.15)

    train_data = processed_entries[:train_size]
    valid_data = processed_entries[train_size : train_size + valid_size]
    test_data = processed_entries[train_size + valid_size :]

    # Make sure test data always includes rationales
    for i in range(len(test_data)):
        if len(test_data[i]) == 2:  # No rationales
            test_data[i].append([])  # Add empty rationales

    return {"train": train_data, "valid": valid_data, "test": test_data}


def write_jsonl_files(jsonl_data, output_dir="."):
    """
    Write data to JSONL files.

    Args:
        jsonl_data: Dictionary with train, valid, and test data
        output_dir: Directory to write files to
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Write train data
    with open(os.path.join(output_dir, "train.jsonl"), "w", encoding="utf-8") as f:
        for entry in jsonl_data["train"]:
            f.write(json.dumps(entry) + "\n")

    # Write valid data
    with open(os.path.join(output_dir, "valid.jsonl"), "w", encoding="utf-8") as f:
        for entry in jsonl_data["valid"]:
            f.write(json.dumps(entry) + "\n")

    # Write test data
    with open(os.path.join(output_dir, "test.jsonl"), "w", encoding="utf-8") as f:
        for entry in jsonl_data["test"]:
            f.write(json.dumps(entry) + "\n")


def main():
    """
    Main function to process the HateXplain dataset.
    """
    parser = argparse.ArgumentParser(
        description="Process HateXplain dataset to JSONL format"
    )
    parser.add_argument(
        "--input", type=str, required=True, help="Path to the input JSON file"
    )
    parser.add_argument(
        "--output_dir", type=str, default=".", help="Directory to write output files"
    )

    args = parser.parse_args()

    # Process the data
    jsonl_data = preprocess_hatexplain_file(args.input)

    # Write to files
    write_jsonl_files(jsonl_data, args.output_dir)

    # Print statistics
    print(f"Processing complete. Files written in {args.output_dir}:")
    print(f"- train.jsonl: {len(jsonl_data['train'])} entries")
    print(f"- valid.jsonl: {len(jsonl_data['valid'])} entries")
    print(f"- test.jsonl: {len(jsonl_data['test'])} entries")


if __name__ == "__main__":
    main()
