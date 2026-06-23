"""Build a clean, visual, English presentation deck for the project.

Design philosophy: very little text per slide, strong visuals (the project's real
figures, big stat cards, coloured section dividers), and the full spoken script in
the speaker notes so Presenter mode shows exactly what to say.

Run::

    pip install python-pptx
    python scripts/build_pptx.py            # -> report/presentation.pptx
"""
from __future__ import annotations

from pathlib import Path

from PIL import Image
from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_SHAPE
from pptx.enum.text import MSO_ANCHOR, PP_ALIGN
from pptx.util import Emu, Inches, Pt

ROOT = Path(__file__).resolve().parents[1]
FIG = ROOT / "results" / "figures"
CM = ROOT / "results" / "confusion_matrices"
OUT = ROOT / "report" / "presentation.pptx"

# ---- Palette ---------------------------------------------------------------
NAVY = RGBColor(0x16, 0x2A, 0x4E)     # deep background blue
BLUE = RGBColor(0x24, 0x52, 0x8A)     # primary blue
ACCENT = RGBColor(0xF2, 0x7A, 0x3D)   # warm orange
TEAL = RGBColor(0x2E, 0x9E, 0x8F)     # supporting teal
GREEN = RGBColor(0x2E, 0x7D, 0x5B)
RED = RGBColor(0xC0, 0x44, 0x32)
INK = RGBColor(0x20, 0x24, 0x2C)      # near-black text
MUTE = RGBColor(0x6B, 0x72, 0x80)     # grey text
PANEL = RGBColor(0xF1, 0xF4, 0xF9)    # light panel
LIGHTBLUE = RGBColor(0xCF, 0xDD, 0xF0)
WHITE = RGBColor(0xFF, 0xFF, 0xFF)

W, H = Inches(13.333), Inches(7.5)


# ---- Low level helpers -----------------------------------------------------
def _fill(shape, color, line=None, line_w=1.0):
    shape.fill.solid(); shape.fill.fore_color.rgb = color
    if line is None:
        shape.line.fill.background()
    else:
        shape.line.color.rgb = line; shape.line.width = Pt(line_w)
    shape.shadow.inherit = False
    return shape


def rect(slide, x, y, w, h, color, rounded=False, line=None, line_w=1.0):
    shp = slide.shapes.add_shape(
        MSO_SHAPE.ROUNDED_RECTANGLE if rounded else MSO_SHAPE.RECTANGLE, x, y, w, h)
    return _fill(shp, color, line, line_w)


def text(slide, x, y, w, h, runs, align=PP_ALIGN.LEFT, anchor=MSO_ANCHOR.TOP, space=2):
    """runs: list of (string, size, color, bold). Each becomes its own paragraph."""
    tb = slide.shapes.add_textbox(x, y, w, h); tf = tb.text_frame
    tf.word_wrap = True; tf.vertical_anchor = anchor
    for i, (txt, sz, col, bold) in enumerate(runs):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.alignment = align; p.space_after = Pt(space); p.space_before = Pt(0)
        r = p.add_run(); r.text = txt
        r.font.size = Pt(sz); r.font.bold = bold; r.font.color.rgb = col; r.font.name = "Calibri"
    return tb


def head(slide, kicker, title):
    """Light content header: small accent kicker, large title, thin accent rule."""
    text(slide, Inches(0.7), Inches(0.5), Inches(12.0), Inches(0.4),
         [(kicker.upper(), 13, ACCENT, True)])
    text(slide, Inches(0.7), Inches(0.85), Inches(12.0), Inches(0.8),
         [(title, 30, BLUE, True)])
    rect(slide, Inches(0.72), Inches(1.62), Inches(1.3), Pt(4), ACCENT)


def bullets(slide, x, y, w, h, items, size=18, color=INK, gap=12):
    tb = slide.shapes.add_textbox(x, y, w, h); tf = tb.text_frame; tf.word_wrap = True
    for i, it in enumerate(items):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.space_after = Pt(gap)
        m = p.add_run(); m.text = "•  "; m.font.size = Pt(size); m.font.bold = True
        m.font.color.rgb = ACCENT
        r = p.add_run(); r.text = it; r.font.size = Pt(size); r.font.color.rgb = color
        r.font.name = "Calibri"
    return tb


def stat(slide, x, y, w, h, big, label, fill=PANEL, fg=BLUE, lab=MUTE):
    card = rect(slide, x, y, w, h, fill, rounded=True)
    tf = card.text_frame; tf.word_wrap = True; tf.vertical_anchor = MSO_ANCHOR.MIDDLE
    tf.margin_top = Inches(0.05); tf.margin_bottom = Inches(0.05)
    p = tf.paragraphs[0]; p.alignment = PP_ALIGN.CENTER
    r = p.add_run(); r.text = big; r.font.size = Pt(32); r.font.bold = True; r.font.color.rgb = fg
    for j, line in enumerate(label.split("\n")):
        p2 = tf.add_paragraph(); p2.alignment = PP_ALIGN.CENTER
        r2 = p2.add_run(); r2.text = line; r2.font.size = Pt(12.5); r2.font.color.rgb = lab
    return card


def chip(slide, x, y, w, h, title, sub, fill=BLUE, fg=WHITE):
    c = rect(slide, x, y, w, h, fill, rounded=True)
    tf = c.text_frame; tf.word_wrap = True; tf.vertical_anchor = MSO_ANCHOR.MIDDLE
    tf.margin_left = Inches(0.12); tf.margin_right = Inches(0.12)
    p = tf.paragraphs[0]; p.alignment = PP_ALIGN.CENTER
    r = p.add_run(); r.text = title; r.font.size = Pt(15); r.font.bold = True; r.font.color.rgb = fg
    if sub:
        p2 = tf.add_paragraph(); p2.alignment = PP_ALIGN.CENTER
        r2 = p2.add_run(); r2.text = sub; r2.font.size = Pt(10.5)
        r2.font.color.rgb = WHITE if fg == WHITE else MUTE
    return c


def figure(slide, path, x, y, max_w, max_h):
    w, h = Image.open(path).size
    sc = min(max_w / w, max_h / h)
    nw, nh = int(w * sc), int(h * sc)
    bx, by = x + (max_w - nw) // 2, y + (max_h - nh) // 2
    rect(slide, bx - Emu(45000), by - Emu(45000), nw + Emu(90000), nh + Emu(90000),
         WHITE, line=RGBColor(0xD8, 0xDE, 0xE8), line_w=1)
    slide.shapes.add_picture(str(path), bx, by, nw, nh)


def pagenum(slide, idx):
    text(slide, Inches(12.3), Inches(7.05), Inches(0.8), Inches(0.35),
         [(str(idx), 11, MUTE, True)], align=PP_ALIGN.RIGHT)


def note(slide, txt):
    slide.notes_slide.notes_text_frame.text = txt.strip()


def blank(prs):
    return prs.slides.add_slide(prs.slide_layouts[6])


def fullbg(slide, color):
    rect(slide, 0, 0, W, H, color)


# ---------------------------------------------------------------------------
def build() -> Path:
    prs = Presentation()
    prs.slide_width, prs.slide_height = W, H
    idx = 0

    def divider(num, title, sub):
        nonlocal idx
        s = blank(prs); fullbg(s, NAVY)
        rect(s, Inches(0.9), Inches(2.55), Inches(0.16), Inches(2.0), ACCENT)
        text(s, Inches(1.25), Inches(2.15), Inches(3.0), Inches(2.4),
             [(num, 96, RGBColor(0x33, 0x4E, 0x7A), True)])
        text(s, Inches(3.7), Inches(2.7), Inches(9.0), Inches(1.8),
             [(title, 40, WHITE, True), (sub, 18, LIGHTBLUE, False)], space=10)
        idx += 1; pagenum(s, idx)
        return s

    # 1. TITLE -------------------------------------------------------------
    s = blank(prs); fullbg(s, NAVY)
    rect(s, 0, Inches(4.5), W, Inches(0.09), ACCENT)
    text(s, Inches(0.9), Inches(1.7), Inches(11.6), Inches(2.6),
         [("Predicting User Sentiment", 46, WHITE, True),
          ("in Software Issue Reports", 46, WHITE, True)], space=2)
    text(s, Inches(0.95), Inches(3.75), Inches(11.5), Inches(0.6),
         [("Classifying real app reviews as negative, neutral or positive", 19, LIGHTBLUE, False)])
    text(s, Inches(0.95), Inches(4.8), Inches(11.5), Inches(1.4),
         [("Yann MASSOUAM     ·     Vénus BAKIKO     ·     Faïçal DIELO", 21, WHITE, True),
          ("ST2MLE  ·  Machine Learning for IT Engineers", 14, LIGHTBLUE, False)], space=8)
    idx += 1; pagenum(s, idx)
    note(s, """
Good morning everyone. We are Yann, Vénus and Faïçal, and we will present our machine
learning project: predicting the sentiment of user reviews of applications, in three
classes, negative, neutral and positive. In about twenty minutes we will show how, from
a simple review text, we build a full pipeline that cleans, transforms, trains and
evaluates several models, and we will explain why a few choices really matter.
""")

    # 2. AGENDA ------------------------------------------------------------
    s = blank(prs); head(s, "Roadmap", "What we will cover")
    items = [("01", "The problem and the data"), ("02", "The processing pipeline"),
             ("03", "Metrics and results"), ("04", "Tuning and conclusion")]
    y = Inches(2.1)
    for num, label in items:
        rect(s, Inches(0.9), y, Inches(0.95), Inches(0.95), BLUE, rounded=True)
        text(s, Inches(0.9), y, Inches(0.95), Inches(0.95), [(num, 26, WHITE, True)],
             align=PP_ALIGN.CENTER, anchor=MSO_ANCHOR.MIDDLE)
        text(s, Inches(2.1), y, Inches(9.5), Inches(0.95), [(label, 22, INK, False)],
             anchor=MSO_ANCHOR.MIDDLE)
        y = Emu(int(y) + int(Inches(1.15)))
    note(s, """
Here is our roadmap. We start with the problem and the data, then we walk through the
pipeline from left to right. We insist on two key ideas: vectorisation, that is turning
text into numbers, and the metrics, because on this project a high accuracy can hide a
model that does not work. We finish with the tuning experiments and the conclusion.
""")

    # 3. DIVIDER 01 --------------------------------------------------------
    divider("01", "The problem and the data", "What we predict, and the reviews we learn from")

    # 4. PROBLEM -----------------------------------------------------------
    s = blank(prs); head(s, "Context", "Why this matters")
    bullets(s, Inches(0.9), Inches(2.05), Inches(7.1), Inches(4.0), [
        "Goal: read a review, predict its sentiment automatically.",
        "Three classes: negative, neutral, positive (supervised learning).",
        "Useful to triage angry reviews, track satisfaction, save time.",
    ], size=20, gap=18)
    stat(s, Inches(8.4), Inches(2.1), Inches(4.2), Inches(1.45), "3 classes",
         "negative / neutral / positive", fill=PANEL, fg=BLUE)
    stat(s, Inches(8.4), Inches(3.75), Inches(4.2), Inches(1.45), "Text in",
         "one label out", fill=PANEL, fg=TEAL)
    note(s, """
The goal is concrete: from the text of a review, automatically predict whether it is
negative, neutral or positive. It is supervised learning, because every training review
already has a label. The practical value is real: on an app that receives thousands of
reviews, we want to spot unhappy users quickly, track satisfaction, and avoid reading
everything by hand.
""")

    # 5. DATA --------------------------------------------------------------
    s = blank(prs); head(s, "Dataset", "Real Facebook reviews")
    figure(s, FIG / "class_distribution.png", Inches(0.7), Inches(1.95), Inches(6.7), Inches(4.7))
    stat(s, Inches(7.8), Inches(2.0), Inches(4.8), Inches(1.35), "5,923", "unique reviews (Kaggle)", fg=BLUE)
    stat(s, Inches(7.8), Inches(3.45), Inches(4.8), Inches(1.35), "64 / 31 / 4.5 %", "positive / negative / neutral", fg=TEAL)
    stat(s, Inches(7.8), Inches(4.9), Inches(4.8), Inches(1.35), "1-5 ★  →  label", "1-2 neg, 3 neu, 4-5 pos", fg=ACCENT)
    note(s, """
We do not use made up data, but real Facebook app reviews downloaded from Kaggle. Each
review has a one to five star rating that we map to sentiment: one or two stars is
negative, three is neutral, four or five is positive. We remove duplicate texts so that
the same review cannot appear in both training and test. About 5,923 reviews remain.
Notice the imbalance: two thirds of the reviews are positive and only four and a half
percent are neutral. This imbalance is at the heart of our analysis.
""")

    # 6. DIVIDER 02 --------------------------------------------------------
    divider("02", "The processing pipeline", "From raw text to a tuned, evaluated model")

    # 7. PIPELINE ----------------------------------------------------------
    s = blank(prs); head(s, "Overview", "The pipeline at a glance")
    steps = ["Raw text", "Preprocess", "Vectorise", "Balance", "Reduce", "Model + tune", "Evaluate"]
    x = Inches(0.55); y = Inches(2.7); bw = Inches(1.62); bh = Inches(1.25); g = Inches(0.18)
    for i, st in enumerate(steps):
        col = BLUE if i % 2 == 0 else TEAL
        b = rect(s, x, y, bw, bh, col, rounded=True)
        tf = b.text_frame; tf.word_wrap = True; tf.vertical_anchor = MSO_ANCHOR.MIDDLE
        p = tf.paragraphs[0]; p.alignment = PP_ALIGN.CENTER
        r = p.add_run(); r.text = st; r.font.size = Pt(13.5); r.font.bold = True; r.font.color.rgb = WHITE
        if i < len(steps) - 1:
            text(s, x + bw - Emu(20000), y, g + Emu(40000), bh, [("›", 22, ACCENT, True)],
                 align=PP_ALIGN.CENTER, anchor=MSO_ANCHOR.MIDDLE)
        x = Emu(int(x) + int(bw) + int(g))
    text(s, Inches(0.9), Inches(4.7), Inches(11.6), Inches(1.6), [
        ("Preprocess once, then cache the vectorisers: the whole study runs in about 8 minutes.", 16, INK, False),
        ("Balancing and reduction touch the training folds only, never the test set.", 16, INK, False),
    ], space=10)
    note(s, """
Here is the full pipeline, read from left to right. We start from raw text, clean it,
turn it into numbers, we can rebalance the classes and reduce the dimension, then we
train a tuned model, and finally we evaluate. Two engineering choices: we clean once
because that step learns nothing, which makes everything faster, and balancing and
reduction only touch the training data, never the test, so our results stay honest.
""")

    # 8. PREPROCESSING -----------------------------------------------------
    s = blank(prs); head(s, "Step 3a", "Cleaning the text")
    bullets(s, Inches(0.9), Inches(2.05), Inches(11.5), Inches(2.4), [
        "Lower case, remove URLs, punctuation and digits, then lemmatise.",
        "Keep negations (not, no, never): they flip the sentiment.",
    ], size=20, gap=16)
    box = rect(s, Inches(0.9), Inches(4.4), Inches(11.5), Inches(1.7), PANEL, rounded=True)
    tf = box.text_frame; tf.word_wrap = True; tf.vertical_anchor = MSO_ANCHOR.MIDDLE
    tf.margin_left = Inches(0.3)
    p = tf.paragraphs[0]; r = p.add_run(); r.text = "Example"
    r.font.size = Pt(13); r.font.bold = True; r.font.color.rgb = ACCENT
    p2 = tf.add_paragraph(); r2 = p2.add_run()
    r2.text = '"Critical issue, the app keeps crashing!!!"   →   "critical issue app keep crash"'
    r2.font.size = Pt(17); r2.font.color.rgb = INK
    note(s, """
Preprocessing follows the classic sequence: lower case, remove punctuation, URLs and
digits, then tokenise, remove common stop words and lemmatise. Two refinements. We keep
negation words, because not good and good have opposite meanings. And lemmatisation
groups word forms, crashing and crash become the same word, which shrinks the
vocabulary. The example at the bottom shows the concrete result.
""")

    # 9. VECTORISATION -----------------------------------------------------
    s = blank(prs); head(s, "Step 3b", "Turning text into numbers")
    rows = [("Bag-of-Words", "counts each word", "sparse"),
            ("TF-IDF", "counts weighted by rarity", "sparse"),
            ("Word2Vec", "embeddings learned on our reviews", "dense"),
            ("Word2Vec pre-trained", "Google News vectors", "dense")]
    y = Inches(2.05)
    for name, desc, kind in rows:
        chip(s, Inches(0.9), y, Inches(4.1), Inches(0.95), name, "", fill=BLUE)
        text(s, Inches(5.2), y, Inches(5.6), Inches(0.95), [(desc, 16, INK, False)], anchor=MSO_ANCHOR.MIDDLE)
        kc = ACCENT if kind == "sparse" else TEAL
        chip(s, Inches(11.0), y, Inches(1.55), Inches(0.95), kind, "", fill=kc)
        y = Emu(int(y) + int(Inches(1.12)))
    note(s, """
Models only understand numbers, so each review is turned into a vector. We compare the
four representations named in the brief. Bag-of-Words counts words. TF-IDF does the same
but values rare, discriminative words. Word2Vec turns each word into a dense vector, and
we use two versions, one trained on our reviews and one pre-trained on Google News, which
is exactly the pre-trained Word2Vec the brief asks for. Note: Multinomial Naive Bayes
needs non negative values, so for the dense embeddings we switch to the Gaussian variant.
""")

    # 10. MODELS -----------------------------------------------------------
    s = blank(prs); head(s, "Step 3c", "Five classifiers compared")
    cards = [("Naive Bayes", "fast, robust, winner", GREEN),
             ("Decision Tree", "readable, needs pruning", BLUE),
             ("Random Forest", "many averaged trees", BLUE),
             ("AdaBoost", "weak on sparse text", RED),
             ("Logistic Reg.", "regularised baseline", TEAL)]
    cw = Inches(2.32); x = Inches(0.7); y = Inches(2.3)
    for name, sub, col in cards:
        chip(s, x, y, cw, Inches(1.7), name, sub, fill=col)
        x = Emu(int(x) + int(cw) + int(Inches(0.1)))
    text(s, Inches(0.9), Inches(4.6), Inches(11.6), Inches(0.9),
         [("All tuned with 5-fold cross-validation and grid search, scored on macro-F1.", 18, INK, False)])
    note(s, """
We compare five classifiers, from the simplest to the richest. Naive Bayes assumes words
are independent, which is false but very effective, and it will be our winner. The
Decision Tree is readable but over fits, hence pruning. The Random Forest averages many
trees. AdaBoost combines shallow stumps, but on sparse text it is weak. Logistic
Regression is a solid regularised baseline. All are tuned with five fold cross validation
and grid search, optimising the macro-F1 that we explain next.
""")

    # 11. DIVIDER 03 -------------------------------------------------------
    divider("03", "Metrics and results", "Reading the numbers correctly")

    # 12. METRICS ----------------------------------------------------------
    s = blank(prs); head(s, "Evaluation", "The metrics, in one slide")
    cards = [("Accuracy", "share of correct predictions"),
             ("Precision", "when it says X, is it right?"),
             ("Recall", "of true X, how many found?"),
             ("Macro-F1", "the three classes, equally")]
    x = Inches(0.7); cw = Inches(2.95)
    for name, sub in cards:
        c = rect(s, x, Inches(2.15), cw, Inches(1.9), PANEL, rounded=True)
        tf = c.text_frame; tf.word_wrap = True; tf.vertical_anchor = MSO_ANCHOR.MIDDLE
        tf.margin_left = Inches(0.15); tf.margin_right = Inches(0.15)
        p = tf.paragraphs[0]; p.alignment = PP_ALIGN.CENTER
        r = p.add_run(); r.text = name; r.font.size = Pt(19); r.font.bold = True; r.font.color.rgb = BLUE
        p2 = tf.add_paragraph(); p2.alignment = PP_ALIGN.CENTER
        r2 = p2.add_run(); r2.text = sub; r2.font.size = Pt(13); r2.font.color.rgb = MUTE
        x = Emu(int(x) + int(cw) + int(Inches(0.15)))
    band = rect(s, Inches(0.7), Inches(4.5), Inches(11.95), Inches(1.5), BLUE, rounded=True)
    tf = band.text_frame; tf.word_wrap = True; tf.vertical_anchor = MSO_ANCHOR.MIDDLE
    tf.margin_left = Inches(0.3); tf.margin_right = Inches(0.3)
    p = tf.paragraphs[0]; p.alignment = PP_ALIGN.CENTER
    r = p.add_run()
    r.text = "On an imbalanced dataset, watch the macro-F1 and the per class recall, not just the accuracy."
    r.font.size = Pt(18); r.font.bold = True; r.font.color.rgb = WHITE
    note(s, """
Four words to master. Accuracy is the overall share of correct answers, but it misleads
when one class dominates. Precision answers: when the model says a class, is it right.
Recall answers: among the true cases of a class, how many are found. F1 combines the two.
And macro-F1 averages the three per class F1 scores equally, so missing the rare class is
expensive. That is why we tune on macro-F1. Remember the sentence in the blue band.
""")

    # 13. RESULTS ----------------------------------------------------------
    s = blank(prs); head(s, "Results", "Model comparison")
    figure(s, FIG / "comparison_f1.png", Inches(0.6), Inches(1.95), Inches(8.3), Inches(4.8))
    stat(s, Inches(9.3), Inches(2.2), Inches(3.4), Inches(1.8), "0.559",
         "best macro-F1\nTF-IDF + Naive Bayes", fill=GREEN, fg=WHITE, lab=WHITE)
    bullets(s, Inches(9.25), Inches(4.35), Inches(3.6), Inches(2.2), [
        "Top 5 are all TF-IDF / BoW.",
        "AdaBoost comes last.",
    ], size=15, gap=10)
    note(s, """
Here is the comparison of all combinations. The best is TF-IDF with Naive Bayes, at
0.559 macro-F1. Two observations: the top five are all count based methods, which makes
sense on short reviews, and AdaBoost is last because its stumps only see one word at a
time out of thousands. The scores are tight, around 0.5, because every model hits the
same difficulty, the tiny neutral class.
""")

    # 14. ACCURACY TRAP ----------------------------------------------------
    s = blank(prs); head(s, "Key insight", "The accuracy trap")
    figure(s, CM / "TF-IDF_NaiveBayes.png", Inches(0.6), Inches(1.95), Inches(6.0), Inches(4.8))
    stat(s, Inches(7.1), Inches(2.1), Inches(5.5), Inches(1.7), "0.84",
         "accuracy  (looks great)", fill=PANEL, fg=GREEN)
    stat(s, Inches(7.1), Inches(3.95), Inches(5.5), Inches(1.7), "0.000",
         "neutral recall  (never predicted!)", fill=RGBColor(0xFB, 0xE9, 0xE6), fg=RED, lab=RED)
    note(s, """
This slide is the heart of our message. Our best model shows 84 percent accuracy, which
looks excellent. But look at the confusion matrix: the neutral column is completely
empty. The model never predicts the neutral class. The 53 truly neutral reviews are all
sent to negative or positive. The macro-F1 drops to 0.56 and reveals the problem. The
lesson: on imbalanced data, accuracy lies, you must look at the macro-F1 and the per
class recall.
""")

    # 15. IMBALANCE --------------------------------------------------------
    s = blank(prs); head(s, "Key result", "What SMOTE fixes")
    text(s, Inches(0.9), Inches(2.0), Inches(7.0), Inches(0.4),
         [("Strategy             Accuracy        Neutral recall", 14, MUTE, True)])
    rowsd = [("No resampling", "0.836", "0.000", RED),
             ("SMOTE", "0.727", "0.189", TEAL),
             ("Under-sampling", "0.601", "0.472", GREEN)]
    y = Inches(2.5)
    for name, acc, rec, col in rowsd:
        rect(s, Inches(0.9), y, Inches(7.0), Inches(0.85), PANEL, rounded=True)
        text(s, Inches(1.1), y, Inches(3.3), Inches(0.85), [(name, 16, INK, True)], anchor=MSO_ANCHOR.MIDDLE)
        text(s, Inches(4.2), y, Inches(1.7), Inches(0.85), [(acc, 16, MUTE, False)], anchor=MSO_ANCHOR.MIDDLE, align=PP_ALIGN.CENTER)
        text(s, Inches(6.0), y, Inches(1.7), Inches(0.85), [(rec, 18, col, True)], anchor=MSO_ANCHOR.MIDDLE, align=PP_ALIGN.CENTER)
        y = Emu(int(y) + int(Inches(1.0)))
    stat(s, Inches(8.4), Inches(2.5), Inches(4.2), Inches(2.85), "0.00 → 0.47",
         "neutral recall recovered\nby resampling", fill=NAVY, fg=ACCENT, lab=LIGHTBLUE)
    note(s, """
Now the imbalance. Without any treatment, the neutral recall is exactly zero: the
minority is ignored. SMOTE creates synthetic neutral examples and lifts that recall from
zero to 0.19. Under-sampling goes further, to 0.47, but by throwing away data. So
resampling trades majority accuracy for real recognition of the rare class. This is
exactly what we want on this problem.
""")

    # 16. DIVIDER 04 -------------------------------------------------------
    divider("04", "Tuning and conclusion", "Reduction, pruning, early stopping")

    # 17. PCA --------------------------------------------------------------
    s = blank(prs); head(s, "Tuning", "Dimensionality reduction")
    figure(s, FIG / "pca_f1.png", Inches(0.7), Inches(1.95), Inches(7.3), Inches(4.8))
    stat(s, Inches(8.4), Inches(2.3), Inches(4.2), Inches(1.8), "300 dims",
         "keep ~99% of the score\nfrom 4,600 features", fg=BLUE)
    bullets(s, Inches(8.35), Inches(4.4), Inches(4.3), Inches(2.0), [
        "TruncatedSVD, the PCA for sparse text.",
        "Big speed and memory win.",
    ], size=15, gap=10)
    note(s, """
Dimensionality reduction. Our TF-IDF matrix has about 4,600 columns. We use TruncatedSVD,
the variant of PCA suited to sparse text. With only 300 components, about six percent of
the columns, we keep almost all the performance. It is an excellent speed and memory
trade off, and it is even necessary before a dense model such as gradient boosting.
""")

    # 18. PRUNING ----------------------------------------------------------
    s = blank(prs); head(s, "Tuning", "Decision tree pruning")
    figure(s, FIG / "pruning_accuracy.png", Inches(0.7), Inches(1.95), Inches(7.3), Inches(4.8))
    stat(s, Inches(8.4), Inches(2.3), Inches(4.2), Inches(1.8), "1,857 → 69",
         "nodes, while test accuracy\nrises 0.76 → 0.79", fg=BLUE)
    bullets(s, Inches(8.35), Inches(4.4), Inches(4.3), Inches(2.0), [
        "Unpruned tree over fits.",
        "Bias-variance, made visible.",
    ], size=15, gap=10)
    note(s, """
Pruning the decision tree. A free tree builds 1,857 nodes and memorises the data, 0.98 in
training but only 0.76 in test, so it over fits. By pruning, we cut the tree to 69 nodes
and the test accuracy rises to 0.79, while the gap between training and test shrinks. If
we prune too much, around 29 nodes, the tree under fits. This is the bias variance trade
off made concrete.
""")

    # 19. EARLY STOPPING ---------------------------------------------------
    s = blank(prs); head(s, "Tuning", "Early stopping in boosting")
    figure(s, FIG / "early_stopping.png", Inches(0.7), Inches(1.95), Inches(7.3), Inches(4.8))
    stat(s, Inches(8.4), Inches(2.3), Inches(4.2), Inches(1.8), "88 / 500",
         "trees used before stopping\nabout 6x less computation", fg=BLUE)
    bullets(s, Inches(8.35), Inches(4.4), Inches(4.3), Inches(2.0), [
        "Stop when validation plateaus.",
        "Guards against over fitting.",
    ], size=15, gap=10)
    note(s, """
Early stopping for gradient boosting, a model that adds trees one by one. We allow up to
500 trees but monitor a validation slice and stop as soon as performance stops improving.
Here it stops after 88 trees out of 500, about six times less computation, with no loss of
test quality and protection against over fitting.
""")

    # 20. CONCLUSION -------------------------------------------------------
    s = blank(prs); head(s, "Wrap-up", "Conclusion")
    bullets(s, Inches(0.9), Inches(2.05), Inches(12.0), Inches(3.0), [
        "Complete, reproducible pipeline on real Facebook reviews.",
        "Best model: TF-IDF + Naive Bayes, macro-F1 0.559.",
        "Accuracy misleads, macro-F1 tells the truth.",
        "SMOTE gives the minority class a voice.",
    ], size=20, gap=16)
    stat(s, Inches(0.9), Inches(5.4), Inches(5.9), Inches(1.2), "macro-F1, not accuracy",
         "the headline lesson", fill=NAVY, fg=ACCENT, lab=LIGHTBLUE)
    stat(s, Inches(7.0), Inches(5.4), Inches(5.6), Inches(1.2), "SMOTE: 0.00 → 0.47",
         "neutral recall recovered", fill=NAVY, fg=ACCENT, lab=LIGHTBLUE)
    note(s, """
To conclude. We built a complete and reproducible pipeline on real Facebook reviews,
compared four text representations and five models, all tuned with cross validation. The
best is TF-IDF with Naive Bayes at 0.559 macro-F1. The main lesson is methodological: on
an imbalanced dataset, accuracy misleads and the macro-F1 tells the truth. SMOTE is what
lets the minority class exist. Thank you, we are ready for your questions.
""")

    # 21. THANK YOU --------------------------------------------------------
    s = blank(prs); fullbg(s, NAVY)
    rect(s, 0, Inches(4.2), W, Inches(0.09), ACCENT)
    text(s, Inches(1.0), Inches(2.6), Inches(11.3), Inches(2.0),
         [("Thank you", 48, WHITE, True), ("Questions?", 24, LIGHTBLUE, False)],
         align=PP_ALIGN.CENTER, space=10)
    text(s, Inches(1.0), Inches(4.6), Inches(11.3), Inches(0.7),
         [("Yann MASSOUAM   ·   Vénus BAKIKO   ·   Faïçal DIELO", 16, WHITE, True)],
         align=PP_ALIGN.CENTER)
    idx += 1; pagenum(s, idx)
    note(s, "Thank you, we will now take your questions. For metric questions, think macro-F1 and per class recall. For imbalance questions, think SMOTE.")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    prs.save(str(OUT))
    return OUT


if __name__ == "__main__":
    out = build()
    print(f"Wrote {out}  ({out.stat().st_size // 1024} KB)")
