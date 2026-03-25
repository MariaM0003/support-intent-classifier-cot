"""
Support Intent Classifier:
Evaluates LLM-based intent detection using Few-Shot prompting and CoT reasoning.
"""

import os
import json

import matplotlib.pyplot as plt
import seaborn as sns
from openai import OpenAI
from dotenv import load_dotenv
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics import classification_report, multilabel_confusion_matrix

load_dotenv()
client = OpenAI()  # Assumes OPENAI_API_KEY is in your .env

# Professional Prompt Engineering
SYSTEM_PROMPT = """You are a specialized Customer Support Intent Classifier.
Your goal is to extract specific user intents from multi-turn dialogues.
Always return valid JSON. Do not include any conversational filler."""

PROMPT_TEMPLATE = """
Analyze the following support dialogue:
"{dialogue}"

1. Decompose the dialogue into discrete user propositions.
2. Map these propositions to one or more of these intents:
   [Technical_Issue, Refund_Request, Account_Inquiry, Order_Status]

Return ONLY a JSON object in this format:
{{
  "analysis": "Brief reasoning for your choices,
   identifying where intents change or resolve.",
  "intents": ["Intent_1", "Intent_2"]
}}
"""


def save_confusion_matrix(y_true_bin, y_pred_bin, classes):
    """Generates a 2x2 grid of heatmaps for multi-label evaluation."""
    mcm = multilabel_confusion_matrix(y_true_bin, y_pred_bin)

    plt.figure(figsize=(12, 8))
    for i, (matrix, class_name) in enumerate(zip(mcm, classes)):
        plt.subplot(2, 2, i + 1)
        sns.heatmap(
            matrix,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=["False", "True"],
            yticklabels=["False", "True"],
        )
        plt.title(f"Intent: {class_name}")
        plt.ylabel("Actual")
        plt.xlabel("Predicted")

    plt.tight_layout()
    plt.savefig("confusion_matrix.png")
    print("\n[Visual] Confusion matrices saved as 'confusion_matrix.png'")


def run_evaluation():
    # Load dataset
    try:
        with open("dataset.json", "r") as f:
            data = json.load(f)
    except FileNotFoundError:
        print("Error: dataset.json not found.")
        return

    y_true = [item["labels"] for item in data]
    y_pred = []

    print(f"Starting evaluation on {len(data)} dialogues...\n")

    for item in data:
        print(f"Processing: {item['text'][:60]}...")

        try:
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {
                        "role": "user",
                        "content": PROMPT_TEMPLATE.format(
                            dialogue=item["text"]),
                    },
                ],
                response_format={"type": "json_object"},
                temperature=0,
            )

            result = json.loads(response.choices[0].message.content)
            print(f"-> Analysis: {result.get('analysis')}")
            y_pred.append(result.get("intents", []))

        except Exception as e:
            print(f"Error: {e}")
            y_pred.append([])

    # Metrics Calculation
    mlb = MultiLabelBinarizer()
    y_true_bin = mlb.fit_transform(y_true)
    y_pred_bin = mlb.transform(y_pred)

    # Print Report
    print("\n" + "=" * 30)
    print("FINAL PERFORMANCE REPORT")
    print("=" * 30)
    print(
        classification_report(
            y_true_bin, y_pred_bin, target_names=mlb.classes_, zero_division=0
        )
    )

    # Generate Heatmaps
    save_confusion_matrix(y_true_bin, y_pred_bin, mlb.classes_)


if __name__ == "__main__":
    run_evaluation()
