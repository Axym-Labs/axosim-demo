"""Goodfire figure replicas rendered with TikZ/PGFPlots, never Matplotlib."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
import shutil
import subprocess

import numpy as np
from PIL import Image
from scipy.stats import spearmanr

from .tikz_plot import FONT_PATH


PAPER = "F6F5F0"
PLOT = "E5ECF6"
INK = "292929"
MUTED = "6E6A61"
BLUE = "4387A8"
GREEN = "65A84F"
ORANGE = "E69635"
MAGENTA = "D93B9F"
PALETTE = (
    "8750C8",
    "50A7A0",
    "CD5A87",
    "6BAA4B",
    "3E83B6",
    "D28A36",
    "AD4C49",
    "5A58B3",
    "55B2C2",
    "B9A23D",
    "6D9B61",
    "C15BAA",
    "498E78",
    "B46B42",
    "7774C9",
    "3F9CAC",
    "A55A87",
    "78A545",
)


def _number(value):
    return format(float(value), ".7g")


def _escape(value):
    substitutions = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
    }
    return "".join(substitutions.get(char, char) for char in str(value))


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _preamble(width, height):
    colors = "\n".join(
        f"\\definecolor{{C{index}}}{{HTML}}{{{color}}}"
        for index, color in enumerate(PALETTE)
    )
    return rf"""\documentclass[border=0pt]{{standalone}}
\usepackage{{fontspec}}
\setmainfont{{InterVariable.ttf}}[Path={FONT_PATH.resolve().parent}/]
\setsansfont{{InterVariable.ttf}}[Path={FONT_PATH.resolve().parent}/]
\renewcommand{{\familydefault}}{{\sfdefault}}
\usepackage{{graphicx}}
\usepackage{{pgfplots}}
\pgfplotsset{{compat=1.18}}
\usetikzlibrary{{arrows.meta,calc,decorations.pathreplacing,positioning}}
\definecolor{{Paper}}{{HTML}}{{{PAPER}}}
\definecolor{{Plot}}{{HTML}}{{{PLOT}}}
\definecolor{{Ink}}{{HTML}}{{{INK}}}
\definecolor{{Muted}}{{HTML}}{{{MUTED}}}
\definecolor{{GoodBlue}}{{HTML}}{{{BLUE}}}
\definecolor{{GoodGreen}}{{HTML}}{{{GREEN}}}
\definecolor{{GoodOrange}}{{HTML}}{{{ORANGE}}}
\definecolor{{GoodMagenta}}{{HTML}}{{{MAGENTA}}}
{colors}
\pgfplotsset{{colormap={{residual}}{{rgb255=(33,102,172) rgb255=(247,247,247) rgb255=(178,24,43)}}}}
\pgfplotsset{{goodfire/.style={{
  axis background/.style={{fill=Plot}},
  axis line style={{draw=gray!72,line width=.45pt}},
  tick style={{draw=gray!72,line width=.4pt}},
  tick label style={{font=\sffamily\fontsize{{9}}{{11}}\selectfont,text=Ink}},
  label style={{font=\sffamily\bfseries\fontsize{{10}}{{12}}\selectfont,text=Ink}},
  title style={{font=\sffamily\bfseries\fontsize{{12}}{{14}}\selectfont,text=Ink,yshift=5pt}},
  grid=major,grid style={{white,line width=.65pt}},
  major tick length=2pt,
  clip=true
}}}}
\begin{{document}}
\begin{{tikzpicture}}[x=.75bp,y=.75bp]
\path[use as bounding box] (0,0) rectangle ({width},{height});
"""


def _compile(output, filename, width, height, body, *, transparent=False):
    """Compile a single exact-size vector figure and export a pixel preview."""
    for program in ("tectonic", "pdftocairo"):
        if not shutil.which(program):
            raise RuntimeError(f"{program} is required for Goodfire TikZ figures")
    output = Path(output)
    source = output / "tikz-source"
    source.mkdir(parents=True, exist_ok=True)
    stem = Path(filename).stem
    tex = source / f"{stem}.tex"
    tex.write_text(
        _preamble(width, height) + body + "\n\\end{tikzpicture}\n\\end{document}\n"
    )
    run = subprocess.run(
        [
            "tectonic",
            "--keep-logs",
            "--outdir",
            str(source.resolve()),
            str(tex.resolve()),
        ],
        capture_output=True,
        text=True,
    )
    if run.returncode:
        (source / f"{stem}-build.txt").write_text(run.stdout + run.stderr)
        raise RuntimeError(f"Tectonic failed for {filename}; see {stem}-build.txt")
    pdf = source / f"{stem}.pdf"
    final_pdf = output / f"{stem}.pdf"
    shutil.copy2(pdf, final_pdf)
    prefix = source / f"{stem}-4x"
    command = [
        "pdftocairo",
        "-png",
        "-singlefile",
        "-scale-to-x",
        str(width * 4),
        "-scale-to-y",
        str(height * 4),
    ]
    if transparent:
        command.append("-transp")
    subprocess.run([*command, str(pdf), str(prefix)], check=True, capture_output=True)
    rendered = Image.open(prefix.with_suffix(".png")).convert("RGBA")
    rendered = rendered.resize((width, height), Image.Resampling.LANCZOS)
    target = output / filename
    if target.suffix.lower() in {".jpg", ".jpeg"}:
        background = Image.new("RGBA", rendered.size, "white")
        background.alpha_composite(rendered)
        background.convert("RGB").save(target, quality=96, subsampling=0)
    else:
        rendered.save(target)
    prefix.with_suffix(".png").unlink()
    pdf.unlink()
    (source / f"{stem}.log").unlink(missing_ok=True)
    return target


def _bounds(values, padding=0.05):
    values = np.asarray(values)
    lo = float(np.nanmin(values))
    hi = float(np.nanmax(values))
    spread = max(hi - lo, 1e-8)
    return lo - spread * padding, hi + spread * padding


def _coordinates(x, y):
    return "\n".join(f"({_number(a)},{_number(b)})" for a, b in zip(x, y))


def _scatter_by_groups(x, y, groups, *, size=0.55, opacity=0.82):
    pieces = []
    unique = list(dict.fromkeys(np.asarray(groups).tolist()))
    for index, group in enumerate(unique):
        mask = np.asarray(groups) == group
        pieces.append(
            rf"\addplot[only marks,mark=*,mark size={size}pt,color=C{index % len(PALETTE)},draw opacity={opacity},fill opacity={opacity}] coordinates {{{_coordinates(np.asarray(x)[mask], np.asarray(y)[mask])}}};"
        )
    return "\n".join(pieces)


def _axis(x, y, width, height, options, plots):
    return (
        rf"\begin{{axis}}[goodfire,at={{({x * 0.75}bp,{y * 0.75}bp)}},anchor=south west,scale only axis,width={width * 0.75}bp,height={height * 0.75}bp,{options}]"
        + plots
        + "\\end{axis}\n"
    )


def _render_tree(manifest, output):
    rows = sorted(manifest["rows"], key=lambda row: (row["source"], *row["label_path"]))
    paths = [tuple([row["source"], *row["label_path"]]) for row in rows]
    angles = np.linspace(0, 2 * np.pi, len(rows), endpoint=False)
    radii = [22, 72, 138, 210, 282, 350]
    nodes = {(): (0.0, radii[0])}
    for depth in range(1, 6):
        grouped = {}
        for index, path in enumerate(paths):
            grouped.setdefault(path[:depth], []).append(index)
        for prefix, indices in grouped.items():
            center = float(
                np.angle(np.mean(np.exp(1j * angles[indices]))) % (2 * np.pi)
            )
            nodes[prefix] = (center, radii[depth])
    first_labels = sorted({path[1] for path in paths})
    label_color = {
        label: index % len(PALETTE) for index, label in enumerate(first_labels)
    }
    cx, cy = 377.5, 419.5
    pieces = ["\\fill[white] (0,0) rectangle (755,839);\n"]
    for prefix, (angle, radius) in sorted(nodes.items(), key=lambda item: len(item[0])):
        if not prefix:
            continue
        parent_angle, parent_radius = nodes[prefix[:-1]]
        color_index = label_color.get(prefix[1] if len(prefix) > 1 else prefix[0], 0)
        p0 = (
            cx + parent_radius * math.cos(parent_angle),
            cy + parent_radius * math.sin(parent_angle),
        )
        p1 = (cx + radius * math.cos(angle), cy + radius * math.sin(angle))
        bend = (
            cx + parent_radius * math.cos(angle),
            cy + parent_radius * math.sin(angle),
        )
        pieces.append(
            rf"\draw[C{color_index},line width=.52pt,line cap=round] ({_number(p0[0])},{_number(p0[1])}) -- ({_number(bend[0])},{_number(bend[1])}) -- ({_number(p1[0])},{_number(p1[1])});"
        )
    return _compile(
        output,
        "species_avg_phylo_tree.png",
        755,
        839,
        "\n".join(pieces),
        transparent=True,
    )


def _render_sampling(manifest, output):
    count = len(manifest["rows"])
    pieces = ["\\fill[white] (0,0) rectangle (2592,950);\n"]
    pieces += [
        r"\node[font=\sffamily\fontsize{30}{35}\selectfont,text=Ink] at (1300,885) {Move natural images across the retina and get activations};",
        r"\node[font=\sffamily\fontsize{27}{32}\selectfont,text=Ink] at (1360,520) {Average over the final 120 ms of each response};",
        r"\node[font=\sffamily\fontsize{30}{35}\selectfont,text=Ink,align=center] at (2350,882) {Image-direction\\embedding vectors};",
    ]
    for y, label, windows, angle in (
        (740, "Image 1", (620, 965, 1420), 22),
        (450, "Image 2", (540, 1110, 1820), 140),
        (160, f"Image {count}", (670, 1260, 1700), 228),
    ):
        pieces.append(
            rf"\node[anchor=east,font=\sffamily\fontsize{{30}}{{35}}\selectfont,text=Ink] at (240,{y}) {{{label}}};"
        )
        pieces.append(
            rf"\filldraw[fill=blue!20,draw=gray!75,line width=.7pt] (340,{y - 16}) rectangle (1985,{y + 16});"
        )
        for x in windows:
            pieces.append(
                rf"\filldraw[fill=orange!12,draw=GoodOrange,line width=2.2pt,rounded corners=2pt] ({x},{y - 21}) rectangle ({x + 145},{y + 21});"
            )
        rad = math.radians(angle)
        ex, ey = 2260 + 120 * math.cos(rad), y + 105 * math.sin(rad)
        pieces.append(rf"\fill[GoodGreen] (2260,{y}) circle (10pt);")
        pieces.append(
            rf"\draw[-{{Stealth[length=18pt,width=12pt]}},GoodGreen,line width=4pt] (2260,{y}) -- ({_number(ex)},{_number(ey)});"
        )
    pieces.append(
        r"\node[font=\sffamily\fontsize{24}{26}\selectfont,text=Ink] at (180,305) {$\vdots$};"
    )
    pieces.append(
        r"\draw[decorate,decoration={brace,amplitude=9pt,mirror},line width=1.3pt] (610,600) -- (1950,600);"
    )
    return _compile(
        output,
        "sampling_and_averaging.png",
        2592,
        950,
        "\n".join(pieces),
        transparent=True,
    )


def _render_raw_umaps(manifest, recording, analysis, output):
    values = analysis["activation_umap_2d"]
    image_index = recording["image_index"].astype(int)
    selected = _sample_indices(np.ones(len(values), dtype=bool), 1400, 41)
    values = values[selected]
    image_index = image_index[selected]
    sources = np.asarray([manifest["rows"][index]["source"] for index in image_index])
    taxonomic_group = np.asarray(
        [
            manifest["rows"][index]["label_path"][
                0 if manifest["rows"][index]["source"] == "inaturalist" else 1
            ]
            for index in image_index
        ]
    )
    body = ["\\fill[Paper] (0,0) rectangle (1360,504);\n"]
    panels = [
        ("Image source", sources),
        ("Taxonomic group", taxonomic_group),
        ("Base photograph", image_index),
    ]
    xlim, ylim = _bounds(values[:, 0]), _bounds(values[:, 1])
    for panel, (title, groups) in enumerate(panels):
        left = 55 + panel * 445
        body.append(
            rf"\fill[white,rounded corners=12pt] ({left - 26},27) rectangle ({left + 365},438);"
        )
        plots = _scatter_by_groups(
            values[:, 0], values[:, 1], groups, size=1.05, opacity=0.88
        )
        options = (
            f"xmin={xlim[0]},xmax={xlim[1]},ymin={ylim[0]},ymax={ylim[1]},"
            f"title={{{_escape(title)}}},title style={{font=\\sffamily\\bfseries\\fontsize{{18}}{{22}}\\selectfont,text=Ink,yshift=7pt}},xtick=\\empty,ytick=\\empty"
        )
        body.append(_axis(left, 65, 335, 335, options, plots))
    return _compile(output, "species_avg-umaps.jpg", 1360, 504, "\n".join(body))


def _render_graph(output):
    pieces = ["\\fill[white] (0,0) rectangle (2774,936);\n"]
    pieces += [
        r"\node[font=\sffamily\fontsize{31}{37}\selectfont,text=Ink,align=center] at (370,790) {Measure the angle\\between embeddings};",
        r"\node[font=\sffamily\fontsize{31}{37}\selectfont,text=Ink] at (1505,790) {Create a nearest-neighbor graph};",
        r"\node[font=\sffamily\fontsize{31}{37}\selectfont,text=Ink] at (1590,125) {Find shortest paths between points};",
    ]
    origin = np.asarray([300.0, 315.0])
    for angle, color in ((48, "GoodGreen"), (79, "GoodBlue")):
        end = origin + np.asarray(
            [190 * math.cos(math.radians(angle)), 190 * math.sin(math.radians(angle))]
        )
        pieces.append(
            rf"\draw[-{{Stealth[length=15pt,width=11pt]}},{color},line width=3pt] ({origin[0]},{origin[1]}) -- ({_number(end[0])},{_number(end[1])});"
        )
    pieces += [
        r"\fill[GoodGreen] (300,315) circle (8pt);",
        r"\draw[gray!70,line width=1.2pt] (368,392) arc[start angle=49,end angle=78,radius=82];",
        r"\node[font=\sffamily\fontsize{52}{58}\selectfont,text=Ink] at (438,474) {$\theta$};",
    ]
    pts = np.asarray(
        [
            [930, 390],
            [1075, 570],
            [1170, 355],
            [1320, 515],
            [1380, 320],
            [1580, 390],
            [1760, 300],
            [1910, 385],
            [2070, 500],
            [2205, 350],
            [2330, 520],
            [2490, 470],
            [2580, 260],
        ],
        float,
    )
    edges = [
        (0, 1),
        (0, 2),
        (1, 2),
        (1, 3),
        (2, 3),
        (2, 4),
        (3, 4),
        (4, 5),
        (5, 6),
        (5, 7),
        (6, 7),
        (7, 8),
        (8, 9),
        (8, 10),
        (9, 10),
        (9, 11),
        (9, 12),
        (10, 11),
        (11, 12),
    ]
    route = {(1, 3), (3, 4), (4, 5), (5, 7), (7, 8), (8, 9), (9, 12)}
    for left, right in edges:
        width = 5 if (left, right) in route else 1.6
        pieces.append(
            rf"\draw[gray!70,line width={width}pt,line cap=round] ({pts[left, 0]},{pts[left, 1]}) -- ({pts[right, 0]},{pts[right, 1]});"
        )
    for index, (x, y) in enumerate(pts):
        color = (
            "GoodBlue"
            if index == 0
            else "GoodGreen"
            if index == 1
            else "GoodOrange"
            if index == 12
            else "gray!58"
        )
        pieces.append(
            rf"\filldraw[fill={color},draw=gray!75,line width=.8pt] ({x},{y}) circle (13pt);"
        )
    return _compile(
        output, "graph_construction.png", 2774, 936, "\n".join(pieces), transparent=True
    )


def _sample_indices(mask, count, seed):
    indices = np.flatnonzero(mask)
    if len(indices) <= count:
        return indices
    return np.random.default_rng(seed).choice(indices, count, replace=False)


def _render_distance(recording, analysis, metrics, output):
    target = analysis["target_angular_distance"]
    cosine = analysis["cosine_similarity"]
    geodesic = analysis["geodesic_distance"]
    train = analysis["train_mask"].astype(bool)
    pair_i, pair_j = analysis["pair_i"], analysis["pair_j"]
    test_pairs = (~train[pair_i]) & (~train[pair_j])
    body = [
        "\\fill[Paper] (0,0) rectangle (1360,750);",
        r"\node[font=\sffamily\bfseries\fontsize{20}{24}\selectfont,text=Ink] at (680,713) {Distance Comparison};",
        r"\fill[white,rounded corners=11pt] (31,34) rectangle (1329,674);",
    ]
    panels = [
        (
            cosine,
            "Cosine Similarity",
            metrics["raw_spearman_cosine_similarity_vs_angular_distance"],
            "Spearman",
        ),
        (
            geodesic,
            "Geodesic Distance",
            metrics["raw_pearson_knn_geodesic_vs_angular_distance"],
            "Pearson",
        ),
    ]
    for panel, (response, ylabel, correlation, label) in enumerate(panels):
        left = 100 + panel * 630
        dev = _sample_indices(~test_pairs, 1400, 70 + panel)
        held = _sample_indices(test_pairs, 600, 80 + panel)
        plots = (
            rf"\addplot[only marks,mark=*,mark size=.55pt,color=GoodBlue,draw opacity=.12,fill opacity=.12] coordinates {{{_coordinates(target[dev], response[dev])}}};"
            + rf"\addplot[only marks,mark=*,mark size=.65pt,color=GoodGreen,draw opacity=.22,fill opacity=.22] coordinates {{{_coordinates(target[held], response[held])}}};"
        )
        xmin, xmax = _bounds(target, 0.02)
        ymin, ymax = _bounds(response, 0.04)
        options = f"xmin={xmin},xmax={xmax},ymin={ymin},ymax={ymax},xlabel={{Angular Distance}},ylabel={{{ylabel}}},title={{{label} Correlation: {correlation:.2f}}}"
        body.append(_axis(left, 95, 500, 500, options, plots))
        if panel == 1:
            body.append(
                r"\fill[GoodGreen] (1100,175) circle (3pt);\node[anchor=west,font=\sffamily\fontsize{7}{9}\selectfont,text=Ink] at (1110,175) {Held-out images};"
            )
    return _compile(output, "manifold-distance.jpg", 1360, 750, "\n".join(body))


def _render_subspace_distance(analysis, metrics, output):
    target = analysis["target_angular_distance"]
    predicted = analysis["learned_distance"]
    train = analysis["train_mask"].astype(bool)
    pair_i, pair_j = analysis["pair_i"], analysis["pair_j"]
    masks = [
        train[pair_i] & train[pair_j],
        (~train[pair_i]) & (~train[pair_j]),
        train[pair_i] ^ train[pair_j],
    ]
    names = ["train-train pairs", "test-test pairs", "train-test pairs"]
    keys = ["train-train", "test-test", "train-test"]
    body = [
        "\\fill[Paper] (0,0) rectangle (1360,590);",
        r"\node[font=\sffamily\bfseries\fontsize{20}{24}\selectfont,text=Ink] at (680,555) {10-dim Learned Space with Image Holdouts};",
        r"\fill[white,rounded corners=11pt] (31,32) rectangle (1329,516);",
    ]
    for panel, (mask, name, key) in enumerate(zip(masks, names, keys)):
        selected = _sample_indices(mask, 1600, 120 + panel)
        color = "GoodBlue" if panel == 0 else "GoodOrange"
        plots = rf"\addplot[only marks,mark=*,mark size=.6pt,color={color},draw opacity=.16,fill opacity=.16] coordinates {{{_coordinates(target[selected], predicted[selected])}}};\addplot[GoodOrange,dashed,line width=1.2pt,domain=0:3.141593,samples=2] {{x}};"
        xmin, xmax = _bounds(target, 0.02)
        ymin, ymax = _bounds(predicted, 0.03)
        correlation = metrics["learned_pair_correlations"][key]
        title = f"{name}"
        options = f"xmin={xmin},xmax={xmax},ymin={ymin},ymax={ymax},xlabel={{Angular Distance}},ylabel={{Angular Distance Predicted}},title={{{title}}}"
        left = 76 + panel * 435
        body.append(_axis(left, 84, 350, 350, options, plots))
        body.append(
            rf"\node[anchor=west,font=\sffamily\fontsize{{7}}{{9}}\selectfont,text=Ink] at ({left + 8},465) {{Correlation: {correlation:.2f}}};"
        )
        body.append(
            rf"\node[anchor=west,font=\sffamily\fontsize{{7}}{{9}}\selectfont,text=Ink] at ({left + 8},449) {{Variance Unexplained: {(1 - metrics['learned_subspace']['variance_explained']) * 100:.1f}\%}};"
        )
    return _compile(output, "subspace-distance.jpg", 1360, 590, "\n".join(body))


def _render_subspace_2d(recording, analysis, output):
    values = analysis["display_learned_umap_2d"]
    direction = recording["direction_degrees"].astype(int)
    selected = _sample_indices(np.ones(len(values), dtype=bool), 3000, 42)
    values = values[selected]
    direction = direction[selected]
    xlim, ylim = _bounds(values[:, 0]), _bounds(values[:, 1])
    plots = _scatter_by_groups(
        values[:, 0], values[:, 1], direction, size=1.05, opacity=0.92
    )
    body = "\\fill[white,rounded corners=10pt] (0,0) rectangle (316,273);" + _axis(
        18,
        17,
        280,
        238,
        f"xmin={xlim[0]},xmax={xlim[1]},ymin={ylim[0]},ymax={ylim[1]},xtick=\\empty,ytick=\\empty",
        plots,
    )
    return _compile(output, "subspace-umap.png", 316, 273, body)


def _render_deviations(analysis, output):
    values = analysis["activation_umap_2d"]
    component = analysis["activation_residual_component"]
    selected = _sample_indices(np.ones(len(values), dtype=bool), 3000, 43)
    values = values[selected]
    component = component[selected]
    variance = float(analysis["residual_explained"]) * 100
    xlim, ylim = _bounds(values[:, 0]), _bounds(values[:, 1])
    meta = "\n".join(
        f"({_number(x)},{_number(y)}) [{_number(m)}]"
        for x, y, m in zip(values[:, 0], values[:, 1], component)
    )
    plots = rf"\addplot[scatter,only marks,mark=*,mark size=1.05pt,draw=none,scatter src=explicit,point meta min={component.min()},point meta max={component.max()},colormap name=residual] coordinates {{{meta}}};"
    body = [
        "\\fill[Paper] (0,0) rectangle (1360,884);",
        r"\fill[white,rounded corners=11pt] (31,34) rectangle (1329,828);",
    ]
    title = f"UMAP colored by SVD Component 0 (explained variance: {variance:.1f}%)"
    options = f"xmin={xlim[0]},xmax={xlim[1]},ymin={ylim[0]},ymax={ylim[1]},xlabel={{UMAP 1}},ylabel={{UMAP 2}},title={{{_escape(title)}}},colorbar,colorbar style={{title={{Component 0}},title style={{font=\\sffamily\\fontsize{{8}}{{10}}\\selectfont}},width=13pt}}"
    body.append(_axis(122, 100, 930, 650, options, plots))
    return _compile(output, "subspace-deviations.jpg", 1360, 884, "\n".join(body))


def _render_statistics(analysis, output):
    from sklearn.decomposition import PCA

    component = PCA(n_components=1, random_state=0).fit_transform(analysis["learned"])[
        :, 0
    ]
    stats = analysis["image_statistics"]
    names = ["Mean luminance", "RMS contrast", "Edge energy"]
    body = [
        "\\fill[Paper] (0,0) rectangle (1360,590);",
        r"\node[font=\sffamily\bfseries\fontsize{20}{24}\selectfont,text=Ink] at (680,555) {Learned Subspace, PCA Component 0};",
        r"\fill[white,rounded corners=11pt] (31,32) rectangle (1329,516);",
    ]
    for panel, name in enumerate(names):
        response = stats[:, panel]
        rho = float(spearmanr(component, response).statistic)
        plots = rf"\addplot[only marks,mark=*,mark size=.8pt,color=GoodBlue,draw opacity=.42,fill opacity=.42] coordinates {{{_coordinates(component, response)}}};"
        xmin, xmax = _bounds(component)
        ymin, ymax = _bounds(response)
        options = f"xmin={xmin},xmax={xmax},ymin={ymin},ymax={ymax},xlabel={{Component 0}},ylabel={{{_escape(name)}}},title={{{_escape(name)} (rho={rho:.2f})}}"
        body.append(_axis(76 + panel * 435, 84, 350, 350, options, plots))
    return _compile(output, "subspace-nucleotide-stats.jpg", 1360, 590, "\n".join(body))


def _render_3d(direction, values, output, filename, width, height):
    direction = np.asarray(direction)
    selected = _sample_indices(np.ones(len(values), dtype=bool), 3000, 44)
    values = values[selected]
    direction = direction[selected]
    plots = []
    for index, angle in enumerate(sorted(set(direction.tolist()))):
        mask = direction == angle
        coords = "\n".join(
            f"({_number(x)},{_number(y)},{_number(z)})" for x, y, z in values[mask]
        )
        plots.append(
            rf"\addplot3[only marks,mark=*,mark size=.85pt,color=C{index},draw opacity=.92,fill opacity=.92] coordinates {{{coords}}};"
        )
    body = "\\fill[white] (0,0) rectangle (%d,%d);" % (width, height) + _axis(
        45,
        35,
        width - 90,
        height - 70,
        "view={38}{18},axis lines=none,xtick=\\empty,ytick=\\empty,ztick=\\empty,grid=none,axis background/.style={fill=none}",
        "\n".join(plots),
    )
    return _compile(output, filename, width, height, body)


def _render_harmonics(analysis, output):
    power = analysis["harmonic_power"]
    coords = _coordinates(np.arange(1, len(power) + 1), power)
    body = [
        "\\fill[Paper] (0,0) rectangle (878,680);",
        r"\fill[white,rounded corners=11pt] (31,32) rectangle (847,630);",
        r"\node[font=\sffamily\bfseries\fontsize{20}{24}\selectfont,text=Ink] at (439,645) {Direction Harmonics};",
    ]
    plots = rf"\addplot[ybar,bar width=25pt,fill=GoodBlue,draw=none] coordinates {{{coords}}};"
    body.append(
        _axis(
            110,
            100,
            650,
            430,
            f"xmin=.5,xmax={len(power) + 0.5},ymin=0,ymax={max(power) * 1.12},xtick={{{','.join(map(str, range(1, len(power) + 1)))}}},xlabel={{Circular Fourier harmonic}},ylabel={{Direction-modulated power}}",
            plots,
        )
    )
    return _compile(output, "direction-harmonics.png", 878, 680, "\n".join(body))


def _retinal_xy(u, v, width, height):
    x = u.astype(float) + v.astype(float) / 2
    y = v.astype(float) * math.sqrt(3) / 2
    x /= np.max(np.abs(x))
    y /= np.max(np.abs(y))
    return (0.5 + 0.44 * x) * width, (0.5 + 0.44 * y) * height


def _render_retinotopic(manifest, recording, analysis, output):
    voltage = recording["voltage_mv"]
    image_index = recording["image_index"].astype(int)
    direction = recording["direction_degrees"].astype(int)
    analysis_direction = recording["direction_degrees"].astype(int)
    u, v = recording["u"], recording["v"]
    selected = [
        row["base_image_index"]
        for row in manifest["rows"]
        if row["source"] == "inaturalist"
    ][:2]
    angles = [0, 90, 180, 270]
    assets = Path(output) / "tikz-source" / "retinotopic-assets"
    assets.mkdir(parents=True, exist_ok=True)
    body = [
        "\\fill[Paper] (0,0) rectangle (2612,970);",
        r"\fill[white,rounded corners=11pt] (28,30) rectangle (2584,940);",
    ]
    panel_w = 400
    panel_h = 400
    for row_number, image in enumerate(selected):
        source_path = Path(manifest["rows"][image]["path"])
        asset_path = assets / f"image-{row_number}.jpg"
        with Image.open(source_path) as source_image:
            source_image.convert("RGB").resize(
                (400, 400), Image.Resampling.LANCZOS
            ).save(asset_path, quality=94)
        base = voltage[image_index == image].mean(0).reshape(8, 721)
        for column, angle in enumerate(angles):
            x0 = 45 + column * 425
            y0 = 510 - row_number * 440
            idx = np.flatnonzero((image_index == image) & (direction == angle))[0]
            response = (voltage[idx].reshape(8, 721) - base)[:4].mean(0)
            xx, yy = _retinal_xy(u, v, panel_w, panel_h)
            threshold = np.quantile(response, 0.55)
            scale = max(float(np.quantile(response, 0.98) - threshold), 1e-7)
            body.append(
                rf"\node[anchor=south west,inner sep=0pt] at ({x0},{y0}) {{\includegraphics[width={panel_w * 0.75}bp,height={panel_h * 0.75}bp]{{{asset_path.resolve()}}}}};"
            )
            for px, py, value in zip(xx, yy, response):
                level = int(np.clip((value - threshold) / scale, 0, 0.999) * 5)
                if level <= 0:
                    continue
                color = ("C0", "C11", "GoodMagenta", "GoodOrange", "orange!35")[
                    level - 1
                ]
                body.append(
                    rf"\fill[{color},opacity=.58] ({_number(x0 + px)},{_number(y0 + panel_h - py)}) circle (3.2pt);"
                )
            rad = math.radians(angle)
            sx = x0 + 42
            sy = y0 + 45
            ex = sx + 58 * math.cos(rad)
            ey = sy + 58 * math.sin(rad)
            body.append(
                rf"\draw[-{{Stealth[length=11pt,width=8pt]}},white,line width=2.7pt] ({sx},{sy}) -- ({_number(ex)},{_number(ey)});"
            )
    values = analysis["display_learned_umap_3d"]
    selected_points = _sample_indices(np.ones(len(values), dtype=bool), 2400, 45)
    values = values[selected_points]
    analysis_direction = analysis_direction[selected_points]
    plots = []
    for index, angle in enumerate(sorted(set(analysis_direction.tolist()))):
        mask = analysis_direction == angle
        coords = "\n".join(
            f"({_number(x)},{_number(y)},{_number(z)})" for x, y, z in values[mask]
        )
        plots.append(
            rf"\addplot3[only marks,mark=*,mark size=.85pt,color=C{index},draw opacity=.9,fill opacity=.9] coordinates {{{coords}}};"
        )
    body.append(
        _axis(
            1770,
            195,
            760,
            600,
            "view={38}{18},axis lines=none,xtick=\\empty,ytick=\\empty,ztick=\\empty,grid=none,axis background/.style={fill=none}",
            "\n".join(plots),
        )
    )
    return _compile(
        output, "retinotopic-vision-manifold.png", 2612, 970, "\n".join(body)
    )


def render_all(config):
    config = Path(config)
    contract = json.loads(config.read_text())
    root = Path(contract["output_root"])
    output = root / "figures"
    if output.exists():
        for path in sorted(output.rglob("*"), reverse=True):
            if path.is_file():
                path.unlink()
            elif path.is_dir() and not any(path.iterdir()):
                path.rmdir()
    output.mkdir(parents=True, exist_ok=True)
    manifest = json.loads((root / "manifest.json").read_text())
    metrics = json.loads((root / "analysis-summary.json").read_text())
    recording = np.load(root / "axosim-optic-flow-voltage.npz")
    analysis = np.load(root / "analysis.npz")
    paths = [
        _render_tree(manifest, output),
        _render_sampling(manifest, output),
        _render_raw_umaps(manifest, recording, analysis, output),
        _render_graph(output),
        _render_distance(recording, analysis, metrics, output),
        _render_subspace_distance(analysis, metrics, output),
        _render_subspace_2d(recording, analysis, output),
        _render_3d(
            np.asarray(
                [
                    manifest["rows"][index]["label_path"][0]
                    for index in recording["image_index"].astype(int)
                ]
            ),
            analysis["activation_umap_3d"],
            output,
            "species_avg_3d_umap.png",
            878,
            680,
        ),
        _render_3d(
            recording["direction_degrees"],
            analysis["display_learned_umap_3d"],
            output,
            "subspace_3d_umap.png",
            866,
            748,
        ),
        _render_deviations(analysis, output),
        _render_statistics(analysis, output),
        _render_harmonics(analysis, output),
        _render_retinotopic(manifest, recording, analysis, output),
    ]
    recording.close()
    analysis.close()
    metadata = {
        "status": "complete",
        "renderer": "Tectonic TikZ/PGFPlots with Poppler raster previews; no Matplotlib",
        "goodfire_reference": "https://www.goodfire.com/research/phylogeny-manifold",
        "files": [
            {
                "name": path.name,
                "sha256": _sha256(path),
                "size": list(Image.open(path).size),
            }
            for path in paths
        ],
        "semantic_substitutions": {
            "tree": "ImageNet WordNet and iNaturalist label provenance; not a neural target",
            "external_distance": "known circular optic-flow angle separation",
            "holdout": "complete natural photographs",
            "activation": "inverse-normalized AxoSim soma voltage in mV",
            "raw_umap": "uncentered soma-voltage responses, with complete photograph identity visible only as an evaluation color",
            "nucleotide_statistics": "natural-image luminance, contrast and edge energy",
            "3d_animation": "fixed-camera vector plot because presentation rotation was explicitly rejected",
        },
        "display_geometry": {
            "fit_samples": metrics["visualization_samples"],
            "base_photographs": metrics["visualization_base_images"],
            "raw_activation_umap": metrics["display_umap_selection"][
                "raw_activation"
            ],
            "learned_subspace_umap": metrics["display_umap_selection"][
                "learned_subspace"
            ],
            "raw_three_panel_render_sample": 1400,
            "single_manifold_render_sample": 3000,
            "subsampling_role": "deterministic display-only point thinning after UMAP; all 12,000 samples are used to fit the embeddings",
        },
    }
    (output / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    return metadata
