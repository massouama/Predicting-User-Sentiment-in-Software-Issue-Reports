"""Synthetic generator for *software issue reports* labelled with sentiment.

Why synthetic?
--------------
Public, ready-to-use sentiment corpora for GitHub / JIRA issues require either
API credentials or downloads from hosts that are unreachable in a sandboxed
environment.  To keep the project fully self-contained and reproducible we
generate a corpus that mirrors the linguistic structure of real issue trackers:

* **Negative** reports describe bugs, crashes and frustration (the majority on
  any tracker -- hence the deliberate class imbalance).
* **Positive** reports are praise, thanks and "works great now" follow-ups.
* **Neutral** reports are questions, feature requests and factual descriptions.

The generator is built from realistic vocabulary banks and sentence templates.
Three design choices make the resulting task a *genuine* learning problem rather
than trivial keyword matching:

1. **Shared software vocabulary** across all classes (the same components,
   versions and actions appear everywhere), so a model must rely on sentiment
   cues, not topic words.
2. **Variable length and composition** -- reports mix one to three clauses.
3. **Label noise** (:data:`config.LABEL_NOISE`) flips a small fraction of labels
   to emulate annotation disagreement and ambiguous text, capping the achievable
   accuracy below 100 %.

Everything is driven by a single NumPy ``Generator`` seeded from
:data:`config.RANDOM_STATE`, so the dataset is byte-for-byte reproducible.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

import config

# ---------------------------------------------------------------------------
# Vocabulary banks
# ---------------------------------------------------------------------------
# Software artefacts referenced by every class -> forces the model to learn
# sentiment rather than topic.
_COMPONENTS: tuple[str, ...] = (
    "login screen", "dashboard", "search bar", "settings page", "mobile app",
    "desktop client", "REST API", "sync engine", "notification system",
    "export feature", "dark mode", "payment module", "user profile",
    "file uploader", "report generator", "navigation menu", "checkout flow",
    "data importer", "audio player", "calendar widget",
)
_VERSIONS: tuple[str, ...] = (
    "the latest release", "version 2.3", "the v4 beta", "the 1.0.7 build",
    "the newest update", "the nightly build", "the stable channel",
)
_ACTIONS: tuple[str, ...] = (
    "after logging in", "when I click save", "on startup", "during sync",
    "while uploading a file", "after the update", "on a fresh install",
    "when switching tabs", "under heavy load", "on my Android device",
    "on Windows 11", "in offline mode",
)

# Sentiment-bearing clauses.  Each pool gives the class its characteristic tone.
_NEGATIVE_CLAUSES: tuple[str, ...] = (
    "keeps crashing", "is completely broken", "freezes every time",
    "throws a cryptic error", "refuses to load", "is painfully slow",
    "corrupted my data", "is totally unusable", "drains the battery",
    "hangs indefinitely", "shows a blank screen", "ignores my settings",
    "lost all my work", "is a frustrating mess", "fails silently",
    "leaks memory until it dies", "logs me out randomly",
)
_POSITIVE_CLAUSES: tuple[str, ...] = (
    "works flawlessly now", "is incredibly fast", "is a huge improvement",
    "looks absolutely beautiful", "saved me so much time", "is rock solid",
    "feels smooth and responsive", "exceeded my expectations",
    "is exactly what I needed", "runs perfectly", "is well designed",
    "is a fantastic addition", "fixed my problem instantly",
)
_NEUTRAL_CLAUSES: tuple[str, ...] = (
    "could use a configuration option", "behaves differently than documented",
    "needs clearer documentation", "supports CSV but not JSON",
    "requires two extra clicks", "depends on the network status",
    "is described in the changelog", "uses the default theme",
    "applies to all accounts", "is mentioned in the README",
)

# Tone-neutral clauses that fit *any* class.  When one of these is used as the
# core sentence (with probability AMBIGUITY_PROB) the sentiment is no longer
# obvious from the main clause, so the label has to be recovered from weaker
# contextual cues -- the realistic ambiguity that separates strong models from
# weak ones.
_AMBIGUOUS_CLAUSES: tuple[str, ...] = (
    "works as expected most of the time", "is okay but could be better",
    "behaves a bit inconsistently", "is fine after a restart",
    "does the job, more or less", "has changed since the last version",
    "takes some getting used to", "is roughly what I expected",
    "could be more intuitive", "seems stable for now",
    "is about the same as before", "depends on how you use it",
)

# Sentence openers that set the register of each class.
_NEGATIVE_OPENERS: tuple[str, ...] = (
    "Terrible experience:", "I am extremely disappointed --", "This is unacceptable.",
    "Bug:", "Critical issue --", "Really frustrated because", "Once again,",
)
_POSITIVE_OPENERS: tuple[str, ...] = (
    "Thank you so much!", "Great job --", "Love it:", "Amazing update!",
    "Just wanted to say", "Kudos to the team,", "Finally,",
)
_NEUTRAL_OPENERS: tuple[str, ...] = (
    "Question:", "Feature request:", "For reference,", "Note:",
    "Steps to reproduce:", "Just to confirm,", "Heads up:",
)

# Optional trailing requests / remarks, again sharing wording across classes.
_NEGATIVE_TAILS: tuple[str, ...] = (
    "Please fix this as soon as possible.", "I might switch to a competitor.",
    "This has been broken for weeks.", "How is this still not resolved?",
)
_POSITIVE_TAILS: tuple[str, ...] = (
    "Keep up the great work!", "Best update in a long time.",
    "Highly recommend it to everyone.", "No complaints at all.",
)
_NEUTRAL_TAILS: tuple[str, ...] = (
    "Is this the intended behaviour?", "Could you point me to the docs?",
    "Adding it as a suggestion.", "Let me know if you need more details.",
)

# Lookup tables keyed by class label keep :func:`_build_report` branch-free.
_CLAUSES = {"negative": _NEGATIVE_CLAUSES, "positive": _POSITIVE_CLAUSES, "neutral": _NEUTRAL_CLAUSES}
_OPENERS = {"negative": _NEGATIVE_OPENERS, "positive": _POSITIVE_OPENERS, "neutral": _NEUTRAL_OPENERS}
_TAILS = {"negative": _NEGATIVE_TAILS, "positive": _POSITIVE_TAILS, "neutral": _NEUTRAL_TAILS}


def _build_report(label: str, rng: np.random.Generator) -> str:
    """Assemble one realistic issue report for ``label``.

    Structure: ``[opener] <component> <clause> [action]. [tail]``.  Optional
    fragments appear with ~50 % probability, giving reports of naturally varying
    length and word order.

    Two realism mechanisms (see :data:`config.AMBIGUITY_PROB` /
    :data:`config.MIXING_PROB`) make the label non-trivial to recover:

    * with ``AMBIGUITY_PROB`` the core clause is a tone-neutral one that fits any
      class, so the sentiment must come from the opener / tail;
    * with ``MIXING_PROB`` the opener or tail is borrowed from a *different*
      class, producing mixed-signal text.
    """
    pick = lambda pool: pool[rng.integers(len(pool))]  # noqa: E731 - tiny local helper
    other = lambda: config.CLASS_NAMES[rng.integers(len(config.CLASS_NAMES))]  # noqa: E731

    parts: list[str] = []

    if rng.random() < 0.6:                       # optional opener (sometimes mixed)
        src = other() if rng.random() < config.MIXING_PROB else label
        parts.append(pick(_OPENERS[src]))

    # Core clause: class-specific, unless we deliberately blur it.
    component = pick(_COMPONENTS)
    if rng.random() < 0.4:                       # sometimes anchor to a version
        component = f"{component} in {pick(_VERSIONS)}"
    if rng.random() < config.AMBIGUITY_PROB:
        clause = pick(_AMBIGUOUS_CLAUSES)        # tone-neutral -> ambiguous report
    else:
        clause = pick(_CLAUSES[label])
    core = f"The {component} {clause}"
    if rng.random() < 0.5:                       # optional contextual action
        core += f" {pick(_ACTIONS)}"
    parts.append(core + ".")

    if rng.random() < 0.45:                      # optional closing remark (sometimes mixed)
        src = other() if rng.random() < config.MIXING_PROB else label
        parts.append(pick(_TAILS[src]))

    return " ".join(parts)


def generate_dataset(
    n_samples: int = config.N_SAMPLES,
    class_balance: dict[str, float] | None = None,
    label_noise: float = config.LABEL_NOISE,
    random_state: int = config.RANDOM_STATE,
) -> pd.DataFrame:
    """Generate a labelled corpus of synthetic software issue reports.

    Parameters
    ----------
    n_samples:
        Total number of reports to produce.
    class_balance:
        Target proportion per class (defaults to the skewed
        :data:`config.CLASS_BALANCE`).  Values are normalised to sum to one.
    label_noise:
        Fraction of labels randomly reassigned after text generation, emulating
        annotation noise.  The *text* keeps its original tone, so these become
        genuinely hard / mislabelled examples.
    random_state:
        Seed for the NumPy generator -> full reproducibility.

    Returns
    -------
    pandas.DataFrame
        Columns ``text`` (str) and ``label`` (one of
        :data:`config.CLASS_NAMES`), shuffled.
    """
    rng = np.random.default_rng(random_state)
    class_balance = class_balance or config.CLASS_BALANCE

    # Convert target proportions into integer counts that sum exactly to
    # ``n_samples`` (the largest class absorbs any rounding remainder).
    labels = list(class_balance)
    weights = np.array([class_balance[k] for k in labels], dtype=float)
    weights /= weights.sum()
    counts = np.floor(weights * n_samples).astype(int)
    counts[int(weights.argmax())] += n_samples - counts.sum()

    # Generate the text for every report from its *true* tone.
    texts: list[str] = []
    true_labels: list[str] = []
    for label, count in zip(labels, counts):
        for _ in range(count):
            texts.append(_build_report(label, rng))
            true_labels.append(label)

    label_arr = np.array(true_labels, dtype=object)

    # Inject label noise: flip a random subset to a *different* class.
    if label_noise > 0:
        n_noisy = int(round(label_noise * len(label_arr)))
        noisy_idx = rng.choice(len(label_arr), size=n_noisy, replace=False)
        all_classes = np.array(config.CLASS_NAMES, dtype=object)
        for i in noisy_idx:
            alternatives = all_classes[all_classes != label_arr[i]]
            label_arr[i] = alternatives[rng.integers(len(alternatives))]

    df = pd.DataFrame({"text": texts, "label": label_arr})

    # Shuffle so the classes are interleaved (important before any split).
    return df.sample(frac=1.0, random_state=random_state).reset_index(drop=True)


def load_or_create_dataset(path=config.DATASET_PATH, **kwargs) -> pd.DataFrame:
    """Return the cached dataset, generating and caching it on first use.

    Caching the CSV guarantees that every downstream script -- exploration,
    training, reporting -- operates on the exact same data.
    """
    path = Path(path)
    if path.exists():
        return pd.read_csv(path)

    config.ensure_directories()
    df = generate_dataset(**kwargs)
    df.to_csv(path, index=False)
    return df


if __name__ == "__main__":
    # Running this module directly (re)generates and caches the dataset.
    config.ensure_directories()
    data = generate_dataset()
    data.to_csv(config.DATASET_PATH, index=False)
    print(f"Generated {len(data)} reports -> {config.DATASET_PATH}")
    print("\nClass distribution:")
    print(data["label"].value_counts())
    print("\nSample reports:")
    for _, row in data.head(6).iterrows():
        print(f"  [{row['label']:>8}] {row['text']}")
