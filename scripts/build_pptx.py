"""Build a polished, presentation-ready PowerPoint for the project.

Design goals (per the brief): a *logical* deck the audience can follow, with real
content and the project's actual figures embedded, not empty text slides. Every
slide carries **speaker notes** containing the spoken pitch, so Presenter mode
shows exactly what to say.

Run::

    pip install python-pptx
    python scripts/build_pptx.py            # -> report/présentation.pptx
"""
from __future__ import annotations

from pathlib import Path

from PIL import Image
from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.text import MSO_ANCHOR, PP_ALIGN
from pptx.util import Emu, Inches, Pt

ROOT = Path(__file__).resolve().parents[1]
FIG = ROOT / "results" / "figures"
CM = ROOT / "results" / "confusion_matrices"
OUT = ROOT / "report" / "presentation.pptx"

# ---- Theme -----------------------------------------------------------------
PRIMARY = RGBColor(0x1F, 0x3B, 0x6E)   # deep blue
ACCENT = RGBColor(0xE2, 0x6D, 0x3C)    # warm orange
LIGHT = RGBColor(0xF3, 0xF6, 0xFB)     # near-white panel
DARK = RGBColor(0x1A, 0x1A, 0x1A)
GREY = RGBColor(0x55, 0x55, 0x55)
WHITE = RGBColor(0xFF, 0xFF, 0xFF)

EMU_W, EMU_H = Inches(13.333), Inches(7.5)


def _solid(shape, color):
    shape.fill.solid()
    shape.fill.fore_color.rgb = color
    shape.line.fill.background()


def _txt(frame, text, size, color=DARK, bold=False, align=PP_ALIGN.LEFT, font="Calibri"):
    frame.word_wrap = True
    p = frame.paragraphs[0]
    p.alignment = align
    r = p.add_run()
    r.text = text
    r.font.size = Pt(size)
    r.font.bold = bold
    r.font.color.rgb = color
    r.font.name = font
    return p


def header(slide, title, kicker=None):
    """Blue top band with the slide title and an optional kicker."""
    band = slide.shapes.add_shape(1, 0, 0, EMU_W, Inches(1.15))
    _solid(band, PRIMARY)
    accent = slide.shapes.add_shape(1, 0, Inches(1.15), EMU_W, Inches(0.06))
    _solid(accent, ACCENT)
    tb = slide.shapes.add_textbox(Inches(0.55), Inches(0.18), Inches(12.2), Inches(0.9))
    tf = tb.text_frame
    tf.vertical_anchor = MSO_ANCHOR.MIDDLE
    if kicker:
        kp = tf.paragraphs[0]
        kr = kp.add_run(); kr.text = kicker.upper()
        kr.font.size = Pt(12); kr.font.bold = True; kr.font.color.rgb = ACCENT
        tp = tf.add_paragraph()
    else:
        tp = tf.paragraphs[0]
    tr = tp.add_run(); tr.text = title
    tr.font.size = Pt(26); tr.font.bold = True; tr.font.color.rgb = WHITE


def footer(slide, idx):
    tb = slide.shapes.add_textbox(Inches(0.4), Inches(7.05), Inches(12.5), Inches(0.35))
    p = tb.text_frame.paragraphs[0]
    r = p.add_run()
    r.text = f"Predicting User Sentiment in Software Issue Reports        ·        {idx}"
    r.font.size = Pt(9); r.font.color.rgb = GREY


def bullets(slide, items, left, top, width, height, size=18, gap=8):
    """Add a list of (text, level) bullets with accent markers."""
    tb = slide.shapes.add_textbox(left, top, width, height)
    tf = tb.text_frame
    tf.word_wrap = True
    for i, item in enumerate(items):
        text, lvl = (item if isinstance(item, tuple) else (item, 0))
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.space_after = Pt(gap)
        p.level = lvl
        marker = p.add_run()
        marker.text = ("●  " if lvl == 0 else "–  ")
        marker.font.size = Pt(size); marker.font.bold = True
        marker.font.color.rgb = ACCENT if lvl == 0 else PRIMARY
        r = p.add_run(); r.text = text
        r.font.size = Pt(size if lvl == 0 else size - 2)
        r.font.color.rgb = DARK
    return tb


def image_fit(slide, path, left, top, max_w, max_h, frame=True):
    """Place an image scaled to fit the (max_w, max_h) box, centred."""
    w, h = Image.open(path).size
    scale = min(max_w / w, max_h / h)
    nw, nh = int(w * scale), int(h * scale)
    x = left + (max_w - nw) // 2
    y = top + (max_h - nh) // 2
    if frame:
        bg = slide.shapes.add_shape(1, x - Emu(40000), y - Emu(40000), nw + Emu(80000), nh + Emu(80000))
        _solid(bg, WHITE); bg.line.color.rgb = RGBColor(0xDD, 0xDD, 0xDD); bg.line.width = Pt(1)
    slide.shapes.add_picture(str(path), x, y, nw, nh)


def panel(slide, left, top, width, height, color=LIGHT):
    sh = slide.shapes.add_shape(1, left, top, width, height)
    _solid(sh, color)
    return sh


def table(slide, rows, left, top, width, height, header_fill=PRIMARY, font=12):
    nr, nc = len(rows), len(rows[0])
    gt = slide.shapes.add_table(nr, nc, left, top, width, height).table
    for c in range(nc):
        for r in range(nr):
            cell = gt.cell(r, c)
            cell.text = str(rows[r][c])
            para = cell.text_frame.paragraphs[0]
            para.alignment = PP_ALIGN.LEFT if c == 0 else PP_ALIGN.CENTER
            run = para.runs[0]
            run.font.size = Pt(font)
            if r == 0:
                run.font.bold = True; run.font.color.rgb = WHITE
                cell.fill.solid(); cell.fill.fore_color.rgb = header_fill
            else:
                run.font.color.rgb = DARK
                cell.fill.solid(); cell.fill.fore_color.rgb = WHITE if r % 2 else LIGHT
    return gt


def notes(slide, text):
    slide.notes_slide.notes_text_frame.text = text.strip()


def blank(prs):
    return prs.slides.add_slide(prs.slide_layouts[6])


# ---------------------------------------------------------------------------
def build() -> Path:
    prs = Presentation()
    prs.slide_width, prs.slide_height = EMU_W, EMU_H
    n = 0

    # 1. TITLE -------------------------------------------------------------
    s = blank(prs)
    bg = s.shapes.add_shape(1, 0, 0, EMU_W, EMU_H); _solid(bg, PRIMARY)
    strip = s.shapes.add_shape(1, 0, Inches(4.55), EMU_W, Inches(0.08)); _solid(strip, ACCENT)
    t = s.shapes.add_textbox(Inches(0.9), Inches(2.0), Inches(11.5), Inches(2.4)).text_frame
    t.word_wrap = True
    _txt(t, "Predicting User Sentiment", 44, WHITE, bold=True)
    p = t.add_paragraph(); r = p.add_run(); r.text = "in Software Issue Reports"
    r.font.size = Pt(44); r.font.bold = True; r.font.color.rgb = WHITE
    p2 = t.add_paragraph(); r2 = p2.add_run()
    r2.text = "Analyse de sentiment d'avis applicatifs : négatif / neutre / positif"
    r2.font.size = Pt(18); r2.font.color.rgb = RGBColor(0xC9, 0xD6, 0xEA)
    names = s.shapes.add_textbox(Inches(0.9), Inches(4.85), Inches(11.5), Inches(1.6)).text_frame
    _txt(names, "Yann MASSOUAM    ·    Vénus BAKIKO    ·    Faïçal DIELO", 20, WHITE, bold=True)
    pc = names.add_paragraph(); rc = pc.add_run()
    rc.text = "ST2MLE  ·  Machine Learning for IT Engineers"
    rc.font.size = Pt(14); rc.font.color.rgb = RGBColor(0xC9, 0xD6, 0xEA)
    notes(s, """
Bonjour a tous. Nous sommes Yann, Vénus et Faïçal, et nous vous présentons notre
projet de Machine Learning : prédire le sentiment d'avis d'utilisateurs sur des
applications, en trois classes, négatif, neutre et positif. Pendant les prochaines
minutes nous allons vous montrer comment, a partir d'un simple texte d'avis, on
construit une chaîne complète qui nettoie, transforme, entraîne et evalue plusieurs
modèles, puis nous expliquerons pourquoi certains choix comptent vraiment.
""")
    n += 1; footer(s, n)

    # 2. AGENDA ------------------------------------------------------------
    s = blank(prs); header(s, "Plan de la présentation", "Fil conducteur")
    bullets(s, [
        "Le problème et pourquoi il est utile",
        "Les données : de vrais avis, un fort déséquilibre",
        "La chaîne de traitement, etape par etape",
        "Comment on transforme le texte en nombres (vectorisation)",
        "Les cinq modèles compares et leur reglage",
        "Les métriques, expliquees simplement",
        "Résultats, le piege de l'accuracy, et ce que SMOTE corrige",
        "Réglages : réduction de dimension, élagage, early stopping",
        "Conclusion, limites et perspectives",
    ], Inches(0.8), Inches(1.5), Inches(11.7), Inches(5.3), size=18, gap=6)
    notes(s, """
Voici notre fil conducteur. On part du problème, puis des données, puis on deroule
la chaîne de traitement de gauche a droite. On insiste sur deux idees fortes :
d'abord la vectorisation, c'est a dire comment passer du texte aux nombres ; ensuite
les métriques, parce que sur ce projet une bonne accuracy peut cacher un modèle qui
ne marche pas. On termine par les reglages classiques et la conclusion.
""")
    n += 1; footer(s, n)

    # 3. PROBLEM -----------------------------------------------------------
    s = blank(prs); header(s, "Le problème et son interet", "1. Contexte")
    bullets(s, [
        "Objectif : classer automatiquement le sentiment d'un avis applicatif.",
        "Trois classes : négatif, neutre, positif (apprentissage supervise).",
        "Entree : le texte d'un avis.  Sortie : une classe.",
        ("Pourquoi c'est utile :", 0),
        ("Trier en priorite les avis negatifs a traiter", 1),
        ("Mesurer la satisfaction et suivre les tendances", 1),
        ("Gagner du temps face a des milliers d'avis", 1),
    ], Inches(0.8), Inches(1.5), Inches(11.7), Inches(5.2), size=19, gap=8)
    notes(s, """
Le but est concret : a partir du texte d'un avis, prédire automatiquement s'il est
négatif, neutre ou positif. C'est de l'apprentissage supervise, car chaque avis
d'entrainement possede deja une etiquette. L'interet pratique est reel : sur une
application qui recoit des milliers d'avis, on veut reperer vite les mecontents,
suivre la satisfaction, et ne pas tout lire a la main.
""")
    n += 1; footer(s, n)

    # 4. DATA --------------------------------------------------------------
    s = blank(prs); header(s, "Les données : de vrais avis Facebook", "2. Données")
    bullets(s, [
        "Avis Facebook reels (Kaggle) : texte + note de 1 a 5 etoiles.",
        "Note vers sentiment : 1-2 = négatif, 3 = neutre, 4-5 = positif.",
        "Apres deduplication du texte : 5 923 avis uniques.",
        ("Déséquilibre naturel et fort :", 0),
        ("positif 64 %  ·  négatif 31 %  ·  neutre 4,5 %", 1),
    ], Inches(0.7), Inches(1.45), Inches(6.6), Inches(5.2), size=17, gap=8)
    image_fit(s, FIG / "class_distribution.png", Inches(7.5), Inches(1.5), Inches(5.4), Inches(5.0))
    notes(s, """
Point important : nous n'utilisons pas de données inventees, mais de vrais avis
Facebook telecharges sur Kaggle. Chaque avis a une note de une a cinq etoiles que
l'on transforme en sentiment : une ou deux etoiles c'est négatif, trois c'est
neutre, quatre ou cinq c'est positif. On supprime les doublons de texte pour eviter
qu'un meme avis se retrouve a la fois en entrainement et en test. Il reste 5 923
avis. Regardez le déséquilibre a droite : deux tiers d'avis positifs et seulement
4,5 pourcent de neutres. Ce déséquilibre sera au coeur de notre analyse.
""")
    n += 1; footer(s, n)

    # 5. PIPELINE ----------------------------------------------------------
    s = blank(prs); header(s, "La chaîne de traitement", "3. Vue d'ensemble")
    steps = ["Texte brut", "Prétraitement", "Vectorisation", "Equilibrage\n(train)",
             "Réduction\n(option)", "Modele\n+ reglage", "Évaluation"]
    x = Inches(0.45); y = Inches(2.6); bw = Inches(1.62); bh = Inches(1.2); gapx = Inches(0.18)
    for i, st in enumerate(steps):
        box = s.shapes.add_shape(1, x, y, bw, bh)
        _solid(box, LIGHT if i % 2 else PRIMARY)
        tf = box.text_frame; tf.word_wrap = True; tf.vertical_anchor = MSO_ANCHOR.MIDDLE
        _txt(tf, st, 12.5, (DARK if i % 2 else WHITE), bold=True, align=PP_ALIGN.CENTER)
        if i < len(steps) - 1:
            ar = s.shapes.add_textbox(x + bw, y, gapx, bh)
            _txt(ar.text_frame, ">", 18, ACCENT, bold=True, align=PP_ALIGN.CENTER)
        x = Emu(int(x) + int(bw) + int(gapx))
    bullets(s, [
        "Pretraiter une seule fois (operation sans apprentissage) puis mettre en cache : etude complète en environ 8 minutes.",
        "L'equilibrage et la réduction n'agissent que sur l'entrainement : aucune fuite vers le test.",
    ], Inches(0.7), Inches(4.4), Inches(12.0), Inches(2.2), size=16, gap=10)
    notes(s, """
Voici la chaîne complète, a lire de gauche a droite. On part du texte brut, on le
nettoie, on le transforme en nombres, on peut rEequilibrer les classes et reduire
la dimension, puis on entraîne un modèle regle, et enfin on evalue. Deux choix
d'ingenierie : on nettoie une seule fois car cette etape n'apprend rien, ce qui
rend tout plus rapide ; et l'equilibrage comme la réduction ne touchent que les
données d'entrainement, jamais le test, pour que nos résultats restent honnetes.
""")
    n += 1; footer(s, n)

    # 6. PREPROCESSING -----------------------------------------------------
    s = blank(prs); header(s, "Prétraitement du texte", "3a. Nettoyage")
    bullets(s, [
        "Minuscules, suppression des URL, de la ponctuation et des chiffres.",
        "Tokenisation, puis suppression des mots vides (stop-words).",
        "On garde les negations (not, no, never) : elles inversent le sentiment.",
        "Lemmatisation : crashing devient crash, issues devient issue.",
    ], Inches(0.7), Inches(1.5), Inches(12.0), Inches(2.6), size=18, gap=10)
    ex = panel(s, Inches(0.7), Inches(4.5), Inches(12.0), Inches(1.9))
    tf = ex.text_frame; tf.word_wrap = True; tf.margin_left = Inches(0.25); tf.vertical_anchor = MSO_ANCHOR.MIDDLE
    _txt(tf, "Exemple", 14, ACCENT, bold=True)
    p = tf.add_paragraph(); r = p.add_run()
    r.text = '"Critical issue -- The app keeps crashing!!!"   devient   "critical issue app keep crash"'
    r.font.size = Pt(16); r.font.color.rgb = DARK
    notes(s, """
Le prétraitement suit la sequence classique : minuscules, retrait de la ponctuation,
des URL et des chiffres, puis tokenisation et suppression des mots vides. Deux
finesses a defendre. D'abord on conserve les negations, car not good et good ont des
sens opposes ; les supprimer detruirait le signal. Ensuite la lemmatisation regroupe
les formes d'un mot, crashing et crash deviennent le meme mot, ce qui reduit le
vocabulaire. L'exemple en bas montre le resultat concret.
""")
    n += 1; footer(s, n)

    # 7. VECTORISATION -----------------------------------------------------
    s = blank(prs); header(s, "Transformer le texte en nombres", "3b. Vectorisation")
    table(s, [
        ["Méthode", "Idée", "Sortie"],
        ["Bag-of-Words", "compte les mots", "creux"],
        ["TF-IDF", "compte, pondere par la rarete du mot", "creux"],
        ["Word2Vec", "vecteurs appris sur nos avis (moyenne)", "dense"],
        ["Word2Vec pre-entraîne", "vecteurs Google News (moyenne)", "dense"],
        ["BERT", "embeddings contextuels (code fourni)", "dense"],
    ], Inches(0.7), Inches(1.5), Inches(12.0), Inches(3.0), font=14)
    bullets(s, [
        "Creux et positif (BoW, TF-IDF) : compatible avec Naive Bayes multinomial.",
        "Dense (embeddings) : on bascule sur Naive Bayes gaussien.",
        "BERT respecte le sujet (option) : execute via un script pret pour Colab.",
    ], Inches(0.7), Inches(4.8), Inches(12.0), Inches(2.0), size=15, gap=7)
    notes(s, """
Les modèles ne comprennent que des nombres, il faut donc transformer le texte. On
compare quatre representations demandees par le sujet. Bag-of-Words compte les mots.
TF-IDF fait pareil mais valorise les mots rares et discriminants. Word2Vec represente
chaque mot par un vecteur dense ; on en a deux versions, une entrainee sur nos avis
et une pre-entrainee sur Google News, qui est exactement le pre-trained Word2Vec du
sujet. BERT est fourni en option. Detail technique utile : Naive Bayes multinomial
exige des valeurs positives, donc pour les embeddings on passe automatiquement a la
version gaussienne.
""")
    n += 1; footer(s, n)

    # 8. MODELS ------------------------------------------------------------
    s = blank(prs); header(s, "Cinq modèles compares", "3c. Classifieurs")
    table(s, [
        ["Modele", "Intuition", "Remarque"],
        ["Naive Bayes", "probabilites des mots par classe", "rapide, robuste, gagnant"],
        ["Arbre de decision", "questions oui/non sur les mots", "a elaguer"],
        ["Forêt aleatoire", "moyenne de nombreux arbres", "robuste"],
        ["AdaBoost", "souches combinees, focus erreurs", "faible sur texte creux"],
        ["Regression logistique", "frontiere lineaire reguliere (L2)", "baseline solide"],
    ], Inches(0.7), Inches(1.5), Inches(12.0), Inches(3.0), font=14)
    bullets(s, [
        "Tous regles par validation croisee 5-fold et GridSearchCV.",
        "Critere de reglage : le macro-F1 (toutes les classes a egalite).",
    ], Inches(0.7), Inches(4.9), Inches(12.0), Inches(1.6), size=16, gap=8)
    notes(s, """
Nous comparons cinq classifieurs, du plus simple au plus riche. Naive Bayes suppose
les mots independants, c'est faux mais tres efficace et ce sera notre gagnant. L'arbre
de decision est lisible mais surapprend, d'ou l'élagage. La forêt aleatoire moyenne
beaucoup d'arbres. AdaBoost combine des souches, mais sur du texte creux il est
faible. La regression logistique est une baseline solide avec regularisation. Tous
sont regles par validation croisee en cinq blocs et recherche sur grille, en
optimisant le macro-F1 que nous expliquons juste apres.
""")
    n += 1; footer(s, n)

    # 9. METRICS -----------------------------------------------------------
    s = blank(prs); header(s, "Les métriques, simplement", "4. Évaluation")
    bullets(s, [
        ("Accuracy : proportion globale de bonnes predictions. Trompeuse si déséquilibre.", 0),
        ("Precision (classe X) : quand le modèle dit X, a-t-il raison ?", 0),
        ("Rappel (classe X) : parmi les vrais X, combien sont retrouves ?", 0),
        ("F1 : moyenne harmonique de precision et rappel.", 0),
        ("Macro-F1 : moyenne des F1 des trois classes, a egalite.", 0),
    ], Inches(0.7), Inches(1.5), Inches(12.0), Inches(3.1), size=18, gap=10)
    box = panel(s, Inches(0.7), Inches(4.9), Inches(12.0), Inches(1.5), color=RGBColor(0xFB, 0xEF, 0xE9))
    tf = box.text_frame; tf.word_wrap = True; tf.margin_left = Inches(0.25); tf.vertical_anchor = MSO_ANCHOR.MIDDLE
    _txt(tf, "A retenir : sur un jeu déséquilibre, on regarde le macro-F1 et le rappel par classe, pas seulement l'accuracy.",
         16, PRIMARY, bold=True)
    notes(s, """
Quatre mots a maitriser. L'accuracy est la proportion globale de bonnes reponses,
mais elle trompe quand une classe domine. La precision repond a : quand le modèle
annonce une classe, a-t-il raison. Le rappel repond a : parmi les vrais cas d'une
classe, combien sont retrouves. Le F1 combine les deux. Et le macro-F1 fait la
moyenne des F1 des trois classes a egalite, donc rater la classe rare coute cher.
C'est pour cela qu'on l'a choisi comme critere. Retenez la phrase en bas.
""")
    n += 1; footer(s, n)

    # 10. RESULTS ----------------------------------------------------------
    s = blank(prs); header(s, "Résultats : comparaison des modèles", "5. Résultats")
    image_fit(s, FIG / "comparison_f1.png", Inches(0.5), Inches(1.4), Inches(7.6), Inches(5.2))
    table(s, [
        ["Meilleur par vectoriseur", "F1"],
        ["TF-IDF + Naive Bayes", "0,559"],
        ["BoW + Naive Bayes", "0,548"],
        ["W2V pre-entraîne + Forêt", "0,542"],
        ["Word2Vec + Rég. log.", "0,537"],
    ], Inches(8.4), Inches(2.0), Inches(4.4), Inches(2.6), font=13)
    bullets(s, [("Top 5 : uniquement TF-IDF / BoW.", 0), ("AdaBoost ferme la marche.", 0)],
            Inches(8.4), Inches(4.9), Inches(4.5), Inches(1.6), size=14, gap=6)
    notes(s, """
Voici la comparaison de toutes les combinaisons. Le meilleur est TF-IDF avec Naive
Bayes, a 0,559 de macro-F1. Le tableau a droite montre le meilleur modèle pour chaque
vectoriseur. Deux observations : les cinq premiers sont tous des methodes par comptage,
TF-IDF ou Bag-of-Words, ce qui est logique sur des avis courts ; et AdaBoost est bon
dernier car ses souches ne voient qu'un mot a la fois parmi des milliers. Les scores
sont serres, autour de 0,5, parce que tous les modèles butent sur la meme difficulte.
""")
    n += 1; footer(s, n)

    # 11. BEST MODEL / CONFUSION -------------------------------------------
    s = blank(prs); header(s, "Le piege de l'accuracy", "5. Lecture fine")
    image_fit(s, CM / "TF-IDF_NaiveBayes.png", Inches(0.5), Inches(1.4), Inches(6.4), Inches(5.2))
    bullets(s, [
        "Meilleur modèle : accuracy 0,84 mais macro-F1 0,56.",
        "La colonne neutre est vide : le modèle ne predit jamais neutre.",
        "Les 53 vrais neutres tombent en négatif ou positif.",
        "Sans le macro-F1, on n'aurait jamais vu ce defaut.",
    ], Inches(7.1), Inches(1.7), Inches(5.7), Inches(4.6), size=17, gap=12)
    notes(s, """
Ce slide est le coeur de notre message. Notre meilleur modèle affiche 84 pourcent
d'accuracy, ce qui parait excellent. Mais regardez la matrice de confusion : la
colonne neutre est entierement vide. Autrement dit, le modèle ne predit jamais la
classe neutre. Les 53 vrais avis neutres sont tous classes en négatif ou positif. Le
macro-F1, lui, chute a 0,56 et revele ce problème. La lecon : sur des données
desequilibrees, l'accuracy ment, il faut regarder le macro-F1 et le rappel par classe.
""")
    n += 1; footer(s, n)

    # 12. IMBALANCE --------------------------------------------------------
    s = blank(prs); header(s, "Le déséquilibre : ce que SMOTE corrige", "5. Résultat clé")
    table(s, [
        ["Stratégie", "Accuracy", "Macro-F1", "Rappel neutre"],
        ["Aucune", "0,836", "0,555", "0,000"],
        ["SMOTE", "0,727", "0,551", "0,189"],
        ["Sous-échantillonnage", "0,601", "0,505", "0,472"],
    ], Inches(0.7), Inches(1.55), Inches(7.2), Inches(2.2), font=13)
    image_fit(s, FIG / "imbalance_f1.png", Inches(8.2), Inches(1.5), Inches(4.6), Inches(3.4))
    bullets(s, [
        "Sans traitement, rappel neutre exactement nul.",
        "SMOTE cree des exemples neutres synthetiques : rappel 0 vers 0,19.",
        "Le sous-échantillonnage va plus loin (0,47) mais jette des données.",
        "Compromis assume : moins d'accuracy, plus de minorite reconnue.",
    ], Inches(0.7), Inches(4.2), Inches(12.0), Inches(2.4), size=15, gap=8)
    notes(s, """
On attaque maintenant le déséquilibre. Sans aucun traitement, le rappel de la classe
neutre est exactement zero : la minorite est ignoree. SMOTE genere des exemples
neutres synthetiques en interpolant entre voisins, et fait remonter ce rappel de zero
a 0,19. Le sous-échantillonnage, qui supprime des exemples majoritaires, va encore
plus loin a 0,47, mais au prix de données jetees et d'une accuracy plus basse. Le
message : le reequilibrage echange de l'accuracy contre une vraie reconnaissance de
la classe rare. C'est exactement ce qu'on veut sur ce problème.
""")
    n += 1; footer(s, n)

    # 13. PCA --------------------------------------------------------------
    s = blank(prs); header(s, "Réduction de dimension (PCA / TruncatedSVD)", "6. Réglages")
    image_fit(s, FIG / "pca_f1.png", Inches(7.0), Inches(1.5), Inches(5.8), Inches(5.0))
    bullets(s, [
        "TF-IDF a environ 4 600 colonnes (le vocabulaire).",
        "TruncatedSVD : equivalent de la PCA adapte au texte creux.",
        "300 composantes (6,5 %) gardent presque toute la performance.",
        "Gain de vitesse et de memoire, indispensable avant un modèle dense.",
    ], Inches(0.7), Inches(1.7), Inches(6.0), Inches(4.6), size=17, gap=12)
    notes(s, """
La réduction de dimension. Notre matrice TF-IDF a environ 4 600 colonnes. On utilise
TruncatedSVD, qui est l'equivalent de la PCA adapte aux données creuses, car la PCA
classique densifierait la matrice. Le resultat est net : avec seulement 300
composantes, soit 6,5 pourcent des colonnes, on conserve quasiment toute la
performance. C'est un excellent compromis vitesse memoire, et c'est meme indispensable
avant d'utiliser un modèle dense comme le gradient boosting.
""")
    n += 1; footer(s, n)

    # 14. PRUNING ----------------------------------------------------------
    s = blank(prs); header(s, "Élagage de l'arbre de decision", "6. Réglages")
    image_fit(s, FIG / "pruning_accuracy.png", Inches(7.0), Inches(1.5), Inches(5.8), Inches(5.0))
    bullets(s, [
        "Arbre non elague : 1 857 noeuds, surapprentissage (train 0,98 / test 0,76).",
        "Meilleur élagage : 69 noeuds, test 0,79.",
        "Elaguer trop (29 noeuds) commence a sous-apprendre.",
        "Compromis biais-variance rendu visible.",
    ], Inches(0.7), Inches(1.7), Inches(6.0), Inches(4.6), size=17, gap=12)
    notes(s, """
L'élagage de l'arbre de decision. Un arbre laisse libre construit 1 857 noeuds et
memorise les données d'entrainement : 0,98 en train mais seulement 0,76 en test, donc
il surapprend. En elaguant par cout-complexite, on reduit l'arbre a 69 noeuds et la
performance de test monte a 0,79, tout en reduisant l'ecart entre train et test. Si
on elague trop, vers 29 noeuds, l'arbre devient trop simple et sous-apprend. C'est
l'illustration concrete du compromis biais-variance.
""")
    n += 1; footer(s, n)

    # 15. EARLY STOPPING ---------------------------------------------------
    s = blank(prs); header(s, "Early stopping (gradient boosting)", "6. Réglages")
    image_fit(s, FIG / "early_stopping.png", Inches(7.0), Inches(1.5), Inches(5.8), Inches(5.0))
    bullets(s, [
        "On autorise jusqu'a 500 arbres ajoutes un par un.",
        "On surveille une part de validation et on arrete quand ca stagne.",
        "Arrêt a 88 arbres sur 500 : environ six fois moins de calcul.",
        "Protege du surapprentissage sans perte de qualite.",
    ], Inches(0.7), Inches(1.7), Inches(6.0), Inches(4.6), size=17, gap=12)
    notes(s, """
Dernier reglage, l'early stopping pour le gradient boosting, un modèle qui ajoute des
arbres un par un. On autorise jusqu'a 500 arbres, mais on surveille une part de
validation et on arrete des que la performance ne progresse plus. Ici l'arret survient
a 88 arbres sur 500 : environ six fois moins de calcul, sans aucune perte de qualite,
et une protection contre le surapprentissage. C'est simple et tres efficace.
""")
    n += 1; footer(s, n)

    # 16. CONCLUSION -------------------------------------------------------
    s = blank(prs); header(s, "Conclusion", "7. Bilan")
    bullets(s, [
        "Chaîne complète et reproductible sur de vrais avis Facebook.",
        "Comparaison de quatre vectoriseurs et cinq modèles, regles par CV.",
        "Meilleur modèle : TF-IDF + Naive Bayes, macro-F1 0,559.",
        "Lecon majeure : l'accuracy trompe, le macro-F1 dit la verite.",
        "SMOTE est decisif pour faire exister la classe minoritaire.",
        "Élagage, réduction et early stopping limitent le surapprentissage.",
    ], Inches(0.8), Inches(1.5), Inches(11.8), Inches(5.0), size=18, gap=10)
    notes(s, """
Pour conclure. Nous avons construit une chaîne complète et reproductible sur de vrais
avis Facebook, compare quatre representations du texte et cinq modèles, tous regles
par validation croisee. Le meilleur est TF-IDF avec Naive Bayes a 0,559 de macro-F1.
La lecon principale est methodologique : sur un jeu déséquilibre, l'accuracy trompe et
c'est le macro-F1 qui dit la verite. SMOTE est ce qui permet a la classe minoritaire
d'exister. Enfin, élagage, réduction de dimension et early stopping maitrisent le
surapprentissage. Merci, nous sommes prets pour vos questions.
""")
    n += 1; footer(s, n)

    # 17. CHALLENGES / FUTURE ----------------------------------------------
    s = blank(prs); header(s, "Limites et perspectives", "7. Ouverture")
    bullets(s, [
        ("Limites :", 0),
        ("Etiquettes derivees des etoiles : un signal approximatif", 1),
        ("Classe neutre minuscule et ambigue (4,5 %)", 1),
        ("Texte court, bruite, parfois multilingue", 1),
        ("Perspectives :", 0),
        ("Activer BERT (script pret pour Colab)", 1),
        ("Ponderation des classes et seuils ajustes", 1),
        ("Collecter des issues GitHub ou JIRA reelles", 1),
    ], Inches(0.8), Inches(1.5), Inches(11.8), Inches(5.0), size=17, gap=6)
    notes(s, """
Soyons honnetes sur les limites. Nos etiquettes viennent des etoiles, c'est un signal
approximatif. La classe neutre est minuscule et ambigue. Et le texte est court, bruite
et parfois dans une autre langue. Cote perspectives, on peut activer BERT grace au
script que nous fournissons, essayer la ponderation des classes et des seuils
ajustes, et collecter de vraies issues GitHub ou JIRA pour completer.
""")
    n += 1; footer(s, n)

    # 18. THANK YOU --------------------------------------------------------
    s = blank(prs)
    bg = s.shapes.add_shape(1, 0, 0, EMU_W, EMU_H); _solid(bg, PRIMARY)
    strip = s.shapes.add_shape(1, 0, Inches(4.3), EMU_W, Inches(0.08)); _solid(strip, ACCENT)
    tf = s.shapes.add_textbox(Inches(1.0), Inches(2.7), Inches(11.3), Inches(2.0)).text_frame
    _txt(tf, "Merci de votre attention", 40, WHITE, bold=True, align=PP_ALIGN.CENTER)
    p = tf.add_paragraph(); r = p.add_run(); r.text = "Questions ?"
    r.font.size = Pt(24); r.font.color.rgb = RGBColor(0xC9, 0xD6, 0xEA); p.alignment = PP_ALIGN.CENTER
    nm = s.shapes.add_textbox(Inches(1.0), Inches(4.6), Inches(11.3), Inches(0.8)).text_frame
    _txt(nm, "Yann MASSOUAM   ·   Vénus BAKIKO   ·   Faïçal DIELO", 16, WHITE, bold=True, align=PP_ALIGN.CENTER)
    notes(s, "Merci, nous repondrons maintenant a vos questions. Pensez au macro-F1 et a SMOTE pour les questions sur les métriques et le déséquilibre.")
    n += 1; footer(s, n)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    prs.save(str(OUT))
    return OUT


if __name__ == "__main__":
    out = build()
    print(f"Wrote {out}  ({out.stat().st_size // 1024} KB, slides built)")
