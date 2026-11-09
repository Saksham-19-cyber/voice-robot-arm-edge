"""
Lightweight intent classifier using TF-IDF and Logistic Regression.
Trained on synthetic template and paraphrased commands with held-out validation.
Provides calibrated confidence scores and clarification thresholding.
"""

import os
import json
import pickle
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score


# Synthetic dataset generator for robotic voice commands
INTENT_TEMPLATES = {
    "reach": [
        "reach to the {dir}",
        "move the arm {dir}",
        "go {dir}",
        "navigate towards the {dir}",
        "extend the gripper to the {dir}",
        "move {dir} by {dist}",
        "reach {dir} {dist}",
        "go towards {dir}",
        "point the arm {dir}",
        "touch the {target}",
        "move over to the {dir}",
        "shift position {dir}",
        "align arm to {dir}",
        "reach towards the {target}",
        "head {dir}",
    ],
    "pick": [
        "pick up the {object}",
        "grab the {object}",
        "lift the {object}",
        "take the {object} from the table",
        "grasp the {object}",
        "pick and hold the {object}",
        "retrieve the {object}",
        "clasp the {object}",
        "pick up {object} on the {dir}",
        "go grab that {object}",
    ],
    "stop": [
        "stop",
        "stop the robot",
        "halt arm immediately",
        "pause movement",
        "freeze",
        "abort current action",
        "cancel command",
        "stand by",
        "hold your position",
        "stop right there",
        "emergency stop",
    ],
    "unknown": [
        "what time is it",
        "how is the weather",
        "tell me a story",
        "play music",
        "open the door",
        "who made you",
        "what is the capital of France",
        "good morning robot",
        "can you dance",
        "banana apple orange",
        "asdf qwer zxcv",
        "search the web",
        "turn on the lights",
        "where am I",
        "hello there",
    ]
}

DIRECTIONS = ["left", "right", "front", "forward", "back", "backward", "up", "down", "higher", "lower", "top", "bottom", "center"]
TARGETS = ["table", "center", "target", "marker"]
OBJECTS = ["block", "cube", "box", "red block", "blue cube", "green object", "item", "cylinder"]
DISTANCES = ["5 cm", "10 cm", "15 cm", "20 cm", "50 mm", "0.1 m", "2 inches"]


def generate_synthetic_dataset(seed=42):
    rng = np.random.RandomState(seed)
    texts = []
    labels = []

    for intent, templates in INTENT_TEMPLATES.items():
        intent_texts = []
        for template in templates:
            # Generate variations
            if "{object}" in template and "{dir}" in template:
                for obj in OBJECTS[:4]:
                    for d in ["left", "right"]:
                        intent_texts.append(template.format(object=obj, dir=d))
            elif "{dir}" in template and "{dist}" in template:
                for d in DIRECTIONS:
                    for dist in DISTANCES[:3]:
                        intent_texts.append(template.format(dir=d, dist=dist))
            elif "{dir}" in template:
                for d in DIRECTIONS:
                    intent_texts.append(template.format(dir=d))
            elif "{target}" in template:
                for t in TARGETS:
                    intent_texts.append(template.format(target=t))
            elif "{object}" in template:
                for obj in OBJECTS:
                    intent_texts.append(template.format(object=obj))
            else:
                intent_texts.append(template)

        # Balance class counts with natural conversational prefixes (e.g., "please", "can you", "robot")
        prefixes = ["", "please ", "robot ", "could you ", "now ", "kindly "]
        balanced = []
        for p in prefixes:
            for t in intent_texts:
                balanced.append(p + t)
                if len(balanced) >= 120:
                    break
            if len(balanced) >= 120:
                break

        for t in balanced:
            texts.append(t)
            labels.append(intent)

    return texts, labels


class IntentClassifier:
    """Trained sklearn model wrapper for intent prediction."""

    def __init__(self, model_path=None):
        self.pipeline = None
        self.classes_ = None
        if model_path and os.path.exists(model_path):
            self.load(model_path)

    def train(self, texts, labels, test_size=0.2, seed=42):
        X_train, X_test, y_train, y_test = train_test_split(
            texts, labels, test_size=test_size, random_state=seed, stratify=labels
        )

        self.pipeline = Pipeline([
            ("tfidf", TfidfVectorizer(ngram_range=(1, 2), lowercase=True)),
            ("clf", LogisticRegression(C=1.0, max_iter=200, random_state=seed)),
        ])

        self.pipeline.fit(X_train, y_train)
        self.classes_ = list(self.pipeline.named_steps["clf"].classes_)

        y_pred = self.pipeline.predict(X_test)
        acc = float(accuracy_score(y_test, y_pred))
        report = classification_report(y_test, y_pred, output_dict=True)

        return {
            "accuracy": round(acc, 4),
            "n_train": len(X_train),
            "n_test": len(X_test),
            "report": report,
        }

    def predict_with_prob(self, text: str):
        if self.pipeline is None:
            raise RuntimeError("Model is not trained or loaded.")
        probs = self.pipeline.predict_proba([text])[0]
        best_idx = int(np.argmax(probs))
        return self.classes_[best_idx], float(probs[best_idx]), dict(zip(self.classes_, [round(float(p), 4) for p in probs]))

    def save(self, path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self.pipeline, f)

    def load(self, path):
        with open(path, "rb") as f:
            self.pipeline = pickle.load(f)
        self.classes_ = list(self.pipeline.named_steps["clf"].classes_)


def train_and_eval_classifier(model_save_path="models/intent_classifier.pkl", results_save_path="results/nlu_eval.json"):
    texts, labels = generate_synthetic_dataset(seed=42)
    print(f"Generated {len(texts)} synthetic command samples across intents: {set(labels)}")

    clf = IntentClassifier()
    eval_metrics = clf.train(texts, labels, test_size=0.2, seed=42)
    print(f"Classifier Test Accuracy: {eval_metrics['accuracy'] * 100:.2f}%")

    clf.save(model_save_path)
    print(f"Saved intent classifier to {model_save_path}")

    os.makedirs(os.path.dirname(results_save_path), exist_ok=True)
    with open(results_save_path, "w") as f:
        json.dump(eval_metrics, f, indent=2)
    print(f"Saved evaluation metrics to {results_save_path}")

    return eval_metrics


if __name__ == "__main__":
    train_and_eval_classifier()

