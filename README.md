# UMAP an Lieferrouten-Kennzahlen – Streamlit-Demo

**[→ Demo live ausprobieren](https://sebastianhanisch-umap-demo.streamlit.app/)**

Fünftes Stück der **Dimensionsreduktion-Linie** der "Konzepte"-Reihe für die Website "Sebastian Hanisch – Operations Research und Machine Learning":
anders als die Fall-Demos im Portfolio (ein Anwendungsfall, mehrere Verfahren im Vergleich) zeigt diese Demo **ein** Verfahren – **UMAP** – an einem wachsenden Beispiel. Vehikel: **dieselben 12
Lieferrouten-Kennzahlen wie in [pca-demo](../pca-demo), [isomap-demo](../isomap-demo), [lle-demo](../lle-demo) und [tsne-demo](../tsne-demo)** (der Generator ist wortgleich kopiert und per Test gegen dessen Ausgabe
eingefroren), erzeugt aus wenigen versteckten Faktoren – dieselbe gekrümmte Fläche, an der PCA scheiterte. t-SNE, Isomap, LLE und PCA stehen als Vergleich daneben.

**Einordnung in die Reihe (die Kanten des Graphen):** UMAP ist die **Fortsetzung von t-SNE** und zugleich eine **Konvergenz** mit der Graph-Idee von Isomap/LLE: ein **Nachbarschaftsgraph** (hier mit unscharfen Gewichten)
plus eine **lokale Kräfte-Optimierung** (Anziehung entlang der Kanten, Abstoßung von Zufallspunkten). Es soll t-SNEs Schwächen beheben – die Demo **misst**, welche dieser Versprechen sich auf diesen Daten halten:
```
pca-demo → isomap-demo | lle-demo | tsne-demo → umap-demo    (Konvergenz: Nachbarschaftsgraph + Kräfte-Optimierung)
umap-demo → PaCMAP | Autoencoder   (weitere Stücke, noch nicht gebaut)
```

| Versprechen gegenüber t-SNE | Ergebnis (300 Touren; Out-of-sample und Stabilität über feste Seeds, sonst Seed 7) |
|---|---|
| Neue Touren einbetten | ✅ `transform`: R² 0.88–0.93 gegen 0.69–0.85 bei der t-SNE-Näherung; Trainings-Touren verschieben sich beim Neu-Rechnen nur um Procrustes 0.03–0.15 |
| Stabilität (Start egal) | ✅ bei q = 3 (mittlere paarweise Abweichung zufälliger Starts, 4 feste Seeds): UMAP 0.02–0.17, t-SNE 0.56–0.76. Bei q = 2 gemischt: UMAP 0.02–0.32, t-SNE 0.24–0.41; einzelne UMAP-Starts weichen stark ab |
| Geschwindigkeit | ✅ bei großem n (n = 600: 1.3 s gegen 4.9 s); bei n = 100–200 ist t-SNE schneller |
| Globale Struktur | ❌ Abstandstreue ferner Paare 0.56 gegen 0.69 (t-SNE) – nicht besser |
| Sonderfahrten / Extreme | ❌ 5 %: R² 0.14 gegen 0.12 (t-SNE), PCA 0.76 – dieselbe Schwäche |

## Was die Demo zeigt

1. **UMAP in Aktion** (Schritt-Slider + Abspielen): **Fuzzy-Nachbarschaft** einer Tour (Gewichte, ρ, σ) → **Fuzzy-Graph** aller Touren und die **Ähnlichkeitskurve** im Zielraum (`min_dist` → a, b) → **Optimierung**
   (Epochen-Schnappschüsse, Schrittlängen von Anziehung/Abstoßung, R² und exakte Kreuzentropie je Schnappschuss) → Ergebnis neben der PCA.
2. **Was UMAP gefunden hat – und die anderen vier Verfahren auf denselben Daten:** fünf Einbettungen, R² der wahren Faktoren, Abstandstreue (gesamt/nah/fern), Trustworthiness, Graph-Komponenten.
3. **📐 Wie stark hängen die Ergebnisse von n_neighbors und min_dist ab?** (live über feste Sweep-Seeds ab 100000, unabhängig vom Demo-Seed) plus Konvergenz, mit Verdict (Graph zerfällt → nicht konvergiert →
   keine globale Struktur → kein Vorteil → UMAP gewinnt).
4. **🆚 Was UMAP gegenüber t-SNE ändert – gemessen:** Abstände nah/fern; Experimente auf Abruf (Knopf): Out-of-sample (`transform` gegen t-SNE-Näherung), Stabilität (sechs Initialisierungen), Rechenzeit.

Regler: Touren, wahre Faktoren q, Krümmung, Rauschen, **Sonderfahrten**, n_neighbors, min_dist, Epochen, Negative-Sample-Rate, Initialisierung (spektral / PCA / zufällig).

Messwerte (Seed 7, 300 Touren, q = 2, n_neighbors 15, min_dist 0.1, 500 Epochen, spektraler Start, wenn nicht anders angegeben; die Presets prüfen sie mit Bändern):

| Situation | Messung |
|---|---|
| Gekrümmte Fläche | R² der wahren Faktoren **0.88** (UMAP) gegen 0.93 (t-SNE), 0.98 (Isomap), 0.96 (LLE), 0.50 (PCA); Trustworthiness 0.98; **Abstandstreue ferner Paare 0.56** gegen 0.69 (t-SNE), 0.91 (Isomap) |
| 5 % Sonderfahrten | UMAP **R² 0.14**, t-SNE 0.12, PCA 0.76, Isomap 0.75, LLE 0.49; Abstandstreue ferner Paare 0.13 gegen 0.95 (PCA) |
| n_neighbors 3 | Graph in **8 Teilen**, 73 von 300 Touren nicht am Hauptteil, R² **0.34**; n_neighbors 2: 74 Teile, R² 0.11 |
| 20 Epochen | Trustworthiness **0.94** statt 0.98 (500 Epochen), Abstandstreue ferner Paare 0.45 statt 0.56, R² 0.86 |
| Rauschen 0.8 | UMAP **R² 0.89** – vor t-SNE (0.85), Isomap (0.84), LLE (0.58), PCA (0.45) |
| Gerade Daten (Krümmung 0) | **kein Vorteil**: R² 0.91 gegen 0.98 (PCA) |
| q = 3 Faktoren | R² 0.58 (UMAP), 0.55 (t-SNE), 0.51 (Isomap), 0.56 (LLE), 0.19 (PCA) |

**n_neighbors** (gekrümmte Daten, 3 feste Seeds × 200 Touren, 200 Epochen): R² 0.06 (2), 0.35 (3), 0.88 (5), 0.89 (8), 0.92 (15), 0.93 (30), 0.93 (60); Graph-Komponenten im Mittel 57 (2), 6.7 (3), 1 ab 5. Anders als bei t-SNEs
Perplexity gibt es nach oben **kein Fenster** – der Einbruch liegt bei sehr kleinen Werten, an denen der Graph zerfällt. **min_dist:** R² 0.91 (0), 0.92 (0.1), 0.93 (0.25), 0.92 (0.5), 0.94 (1); nur bei 1 steigt die Abstandstreue ferner Paare
(0.65 statt 0.56) – `min_dist` ändert vor allem das Aussehen, kaum die Nachbarschaften. Epochen: 20 → 0.86, 50/100/200/500 → 0.88; Negative-Sample-Rate 1/5/15: R² 0.89/0.88/0.90.

**Stabilität** (vier zufällige Starts je Datensatz, Median und Maximum der sechs paarweisen Procrustes-Abstände; feste Seeds 100000–100003, 300 Touren): q = 3 – UMAP Median 0.07 / 0.17 / 0.13 / 0.02 (Maximum ≤ 0.36),
t-SNE 0.56 / 0.74 / 0.76 / 0.63 (Maximum bis 0.94) → UMAP klar stabiler. q = 2 – UMAP 0.32 / 0.08 / 0.17 / 0.02 (Maximum bis **0.67**), t-SNE 0.26 / 0.41 / 0.24 / 0.38 (Maximum ≤ 0.49) → gemischt, UMAP im Median meist besser, aber
einzelne Starts weichen stark ab. Im Demo-Seed 7 lag der Abstand der zufälligen Starts zum spektralen Lauf lokal bei 0.02 / 0.02 / 0.26 / 0.05, auf der CI-Plattform bei 0.46 / 0.40 / 0.02 / 0.42 – die Optimierung ist chaotisch, die Zahlen
hängen von Plattform (LAPACK/BLAS) und Datensatz ab. Bei kleinem n und wenigen Epochen kann es mehr sein (150 Touren, 100 Epochen: bis 0.52).

**Out-of-sample** (letzte 20 % zurückgehalten, 6 feste Seeds): `transform` R² 0.88, 0.93, 0.88, 0.93, 0.92, 0.93; t-SNE-Näherung 0.79, 0.69, 0.74, 0.85, 0.75, 0.82; Verschiebung der Trainings-Touren beim Neu-Rechnen (UMAP)
0.07, 0.03, 0.15, 0.04, 0.12, 0.03. Trainings-Touren als "neu" landen nahe ihrer eigenen Koordinate (maximale Abweichung ≈ 5 % der Einbettungsbreite). Im Demo-Seed 7 liegen beide fast gleich (UMAP 0.87, t-SNE 0.86) – die Streuung
über Seeds ist groß.

**Rechenzeit** (lokale Messung, 500 Epochen bzw. 500 t-SNE-Iterationen, ein Lauf je n): n = 100: UMAP 0.32 s, t-SNE 0.08 s; n = 200: 0.49 s, 0.24 s; n = 400: **0.88 s, 2.3 s**; n = 600: **1.3 s, 4.9 s**. UMAP braucht zusätzlich den Graphen und eine
dichte spektrale Einbettung – der Wechsel liegt zwischen 200 und 400 Touren.

**Die exakte Kreuzentropie sinkt nicht:** UMAP minimiert nominell die fuzzy Kreuzentropie – gemessen über alle Paare liegt sie im Standardfall am Ende (3979) über dem spektralen Start (1954), und auch die Einbettung der
Referenzimplementierung `umap-learn` liegt mit 4501 darüber. Die stichprobenbasierte Abstoßung optimiert faktisch eine andere Größe (Literatur: Damrich & Hamprecht, 2021); die Demo zeigt die Kurve mit dieser Einordnung.

## Modell und Verfahren

- **Generator** (`umap_scenario.py`): wortgleich aus pca-demo; latente Faktoren, 12 Merkmale in 4 Gruppen, Krümmung `κ·B·h(z)`, Rauschen, Sonderfahrten. UMAP arbeitet immer auf z-Werten.
- **UMAP** (`umap_algorithm.py`, numpy, ohne umap-learn, exaktes kNN, n ≤ 600): folgt umap-learn: kNN inklusive der Tour selbst; ρ = Abstand zum nächsten anderen Punkt; σ per Binärsuche auf log₂(n_neighbors) (alle Zeilen gleichzeitig);
  fuzzy Vereinigung; a, b per eigenem Levenberg-Marquardt; spektrale Einbettung (dichte `eigh` der normalisierten Laplace-Matrix), bei nicht zusammenhängendem Graph Rückfall auf zufällig (in der App gemeldet); SGD über die
  Kanten mit Kantenhäufigkeit ∝ Gewicht, Negative-Sample-Rate, Clip ±4, Lernrate linear fallend – **vektorisiert je Epoche** (alle aktiven Kanten gleichzeitig) statt nacheinander wie umap-learn: gleichwertig, nicht bitgleich;
  `transform` für neue Touren wie in umap-learn (feste Trainingspunkte, Epochen/3, Start am gewichteten Mittel).
- **t-SNE / Isomap / LLE** (`umap_tsne.py`, `umap_isomap.py`, `umap_lle.py`): wortgleich aus tsne-demo / isomap-demo / lle-demo kopiert (nur Vergleichsverfahren, feste gute Einstellungen: Perplexity 30 und 500 Iterationen, k = 10 bzw. k = 14 mit Regularisierung 0.01).
- **Auswertung** (`umap_evaluation.py`): R² der wahren Faktoren aus den zwei Koordinaten per quadratischer Regression; Abstandstreue = Pearson-Korrelation der Paarabstände mit den Faktor-Paarabständen, getrennt für die untere (nah) und obere
  Hälfte (fern); Trustworthiness (Venna & Kaski, eigene Implementierung); Procrustes-Abstand; Sweeps, Stabilität, Out-of-sample, Zeitmessung.

## Was nicht funktioniert hat / Grenzen

- **Die Erwartung "UMAP hat bessere globale Struktur als t-SNE" hat sich nicht bestätigt:** ferne Paare 0.56 gegen 0.69 im Standardfall; bei Sonderfahrten brechen beide gleich stark ein. Das Verdict meldet das als "Keine bessere globale Struktur".
- **Kein Fenster für n_neighbors nach oben:** eine "zu große Nachbarschaft" ließ sich auf diesen Daten nicht als Fehlerfall zeigen (R² steigt bis 60 leicht); die Demo behauptet es deshalb nicht.
- **Konvergenz-Prüfung nur über einen Referenzlauf:** die Qualität entsteht in den letzten Epochen (die Lernrate fällt gegen 0), Zwischenstände sind kein guter Konvergenzindikator – bei wenigen Epochen rechnet die Demo deshalb einen Lauf mit 500
  Epochen daneben.
- **Erste Fassung der Stabilitäts-Aussage war schief:** verglichen wurden UMAP bei q = 2 (Procrustes 0.00–0.26, ein Datensatz) mit t-SNE bei q = 3 (0.46–0.81) – ungleiche Fälle, und auf der CI-Plattform lag der UMAP-Wert bei 0.46. Sauber verglichen (gleiche Datensätze, paarweise, 4 Seeds) gilt der Vorteil nur bei q = 3.
- **Nicht bitgleich zu umap-learn:** Jacobi-artige Aktualisierung; Kurve (a, b), Graph (1e-6) und Größenordnung der Einbettungsqualität stimmen überein (R² 0.88 gegen 0.91), aber Einzelpunkte nicht.
- **Grenzen (Text):** Achsen und Abstände im Bild haben keine feste Bedeutung; Clustergrößen und Abstände zwischen Clustern sollten nicht interpretiert werden. Exaktes kNN und dichte spektrale Einbettung sind auf ≈ 600 Touren
  ausgelegt; größere Datensätze brauchen Nearest-Neighbor-Descent und sparse Eigenlöser, die hier nicht gebaut sind.

## Verifikation

- Kurve (a, b) gegen `umap.umap_.find_ab_params` und `scipy.optimize.curve_fit` (1e-3); Fuzzy-Graph, ρ und σ gegen `umap.umap_.fuzzy_simplicial_set` (Graph 1e-4); kNN gegen `sklearn`; σ-Summe = log₂ k, ρ = Abstand zum nächsten Nachbarn.
- Anziehungs- und Abstoßungsschritt als Gradienten von −log v(d) bzw. −log(1 − v(d)) per finite Differenzen; Optimierung deterministisch bei festem Seed; Schnappschüsse; zerfallener Graph → Zufallsstart und Meldung.
- Qualität gegen `umap-learn` (R² und Trustworthiness vergleichbar); die Aussage "exakte Kreuzentropie sinkt nicht" auch für umap-learn belegt; `transform`: Trainings-Touren als "neu" nahe ihrer Koordinate, deterministisch.
- Generator bit-identisch zu pca-demo, Isomap-/LLE-/t-SNE-Kopien gegen eingefrorene Referenzwerte; Trustworthiness gegen `sklearn.manifold.trustworthiness` (1e-9).
- Sweeps über feste Seeds deterministisch; Verdict-Codes; die Tabelle "Versprechen gegenüber t-SNE" ist als Tests hinterlegt (Sonderfahrten, globale Struktur, Out-of-sample, Stabilität, Rechenzeit-Wechsel); alle 6 Presets in
  kalibrierten Bändern; AppTest-Rauchtests (Default, jedes Preset, jeder Schritt, Randgrößen, n_neighbors folgt n, Experimente auf Abruf), Achsensperre aller Figuren.

## Dateistruktur

| Datei | Zweck |
|---|---|
| `app.py` | Streamlit-App: Schritte, Ergebnis, 📐 n_neighbors/min_dist, 🆚 Vergleich mit t-SNE, Mathe |
| `umap_algorithm.py` | UMAP von Grund auf (Graph, Kurve, Initialisierung, Optimierung, Transform) |
| `umap_tsne.py`, `umap_isomap.py`, `umap_lle.py` | Vergleichsverfahren (wortgleich aus tsne-demo / isomap-demo / lle-demo) |
| `umap_scenario.py`, `umap_constants.py` | Lieferrouten-Generator (wortgleich aus pca-demo), Konstanten, Presets |
| `umap_evaluation.py` | Kennzahlen, Sweeps, Verdict, Stabilität, Out-of-sample, Zeitmessung |
| `umap_presets.py`, `umap_visualization.py` | Permalink/Presets, Plotly-Figuren (achsengesperrt) |
| `tests/` | umap-learn-/sklearn-/scipy-Kreuzvergleiche, Generator-Referenz, Auswertung, Presets, AppTest |

## Lokal ausführen

```bash
python3 -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate

pip install -r requirements.txt
streamlit run app.py
```

## Tests ausführen

```bash
pip install -r requirements-dev.txt
pytest tests/ -v
```

---

Teil des [Operations-Research-Demo-Portfolios](https://sebastianhanisch.net/demos.html) von
[Sebastian Hanisch](https://sebastianhanisch.net) – Operations Research und Machine Learning.
Interesse an einer maßgeschneiderten Lösung? [Kontakt aufnehmen](https://sebastianhanisch.net/kontakt.html).
