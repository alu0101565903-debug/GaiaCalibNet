"""Shared PlotNeuralNet primitives and self-contained diagram export.

Use the same translucent 3D boxes, activation bands and arrows as the library's
VGG16 example. Exported TikZ sources include the original library layer assets.
"""
from __future__ import annotations

from contextlib import redirect_stdout
import hashlib
import io
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile

REPO_ROOT = Path(__file__).resolve().parents[3]
PLOT_NEURAL_NET_ROOT = REPO_ROOT / "PlotNeuralNet"
if not (PLOT_NEURAL_NET_ROOT / "pycore" / "tikzeng.py").is_file():
    raise FileNotFoundError(f"PlotNeuralNet was not found at {PLOT_NEURAL_NET_ROOT}")
sys.path.insert(0, str(PLOT_NEURAL_NET_ROOT))
sys.path.insert(0, str(REPO_ROOT))

from pycore.tikzeng import to_begin, to_connection, to_cor, to_end, to_generate, to_head


def block(name, position, *, width, height, depth, channels="", length="",
           color=r"\ConvColor", band=None):
    """Use the same Box/RightBandedBox pics and edge labels as VGG16.

    Feature-map depth represents wavelength bins, width represents channels;
    the unlabelled height is schematic (these are 1D convolutions).
    """
    kind = "RightBandedBox" if band else "Box"
    options = [
        f"name={name}", f"fill={color}",
        f"height={height}", f"width={width}", f"depth={depth:.3f}",
        'xlabel={{"' + str(channels) + '",""}}', "zlabel=",
        "caption=", "opacity=0.6",
    ]
    if band:
        options.extend([f"bandfill={band}", "bandopacity=0.8"])
    layer = r"\pic at " + position + " {" + kind + "={" + ",".join(options) + "}};\n"
    if length != "":
        # The stock zlabel sits at the near corner and collides with channel counts
        # on narrow 1D blocks. Centre it on the same depth edge instead.
        layer += rf"""
\path ({name}-nearsoutheast) -- node[midway,below=2pt,sloped,font=\small]
    {{{length}}} ({name}-farsoutheast);
"""
    return layer


def caption(name, title, detail=""):
    """Align captions below the projected faces, leaving arrows unobstructed."""
    text = rf"\textbf{{{title}}}"
    if detail:
        text += rf"\\[4pt]{{\small {detail}}}"
    return rf"""
\node[anchor=north,align=center,text width=2.9cm]
    at ({name}-anchor |- caption-row) {{{text}}};
"""


def operation(name, text, x, y, *, width=2.6, fill="white"):
    return rf"""
\node[operation,text width={width}cm,fill={fill}] ({name}) at ({x},{y}) {{{text}}};
\coordinate ({name}-west) at ({name}.west);
\coordinate ({name}-east) at ({name}.east);
\coordinate ({name}-north) at ({name}.north);
\coordinate ({name}-south) at ({name}.south);
\coordinate ({name}-anchor) at ({name}.center);
"""


def add_publication_args(parser):
    """Publication presets use physical dimensions rather than scaled-down fonts."""
    parser.add_argument("--layout", choices=("single-column", "double-column"),
                        default="single-column", help="85 mm (one column) or 178 mm (two columns).")
    parser.add_argument("--font-size", type=float, default=9,
                        help="Label size in points at the PDF's native width (default: 9).")


def publication_architecture(title, nodes, *, layout="single-column", font_size=9, auxiliary=None):
    """Render an ordered network as a compact, alternating-row PlotNeuralNet figure.

    Captions contain the explanatory prose. Shapes in this drawing omit batch;
    all labels retain the requested point size at the native column width.
    """
    import math

    if layout not in ("single-column", "double-column"):
        raise ValueError("Unknown publication layout")
    if not math.isfinite(font_size) or font_size <= 0:
        raise ValueError("font_size must be finite and positive")
    if not nodes:
        raise ValueError("The diagram must contain at least one node")
    width, columns = (8.5, 3) if layout == "single-column" else (17.8, 6)
    rows = math.ceil(len(nodes) / columns)
    row_pitch = 3.7 * max(1, font_size / 9)
    bottom = -(rows - 1) * row_pitch - 2.25 * max(1, font_size / 9)
    palette = {"data": "DataColor", "conv": r"\ConvColor", "pool": r"\PoolColor",
               "dense": r"\FcColor", "dropout": "DropoutColor"}
    layers = [to_head("plotneuralnet").replace("border=8pt", "border=0pt"), r"""
\usepackage{amsmath,amssymb}
\usetikzlibrary{calc,arrows.meta}
""", to_cor(), to_begin(), rf"""
\definecolor{{DataColor}}{{HTML}}{{9DB6C4}}
\definecolor{{DropoutColor}}{{HTML}}{{75B5A1}}
\tikzset{{every node/.style={{font=\fontsize{{{font_size:g}}}{{{font_size * 1.2:g}}}\selectfont}},
    publication flow/.style={{-{{Stealth[length=1.5mm]}},draw=\edgecolor,line width=0.55pt,rounded corners=2pt}},
    publication operation/.style={{draw=black!60,rounded corners=2pt,fill=white,
        align=center,inner sep=3pt}},
    publication label/.style={{anchor=north,align=center,inner sep=1pt,text width=2.55cm}}}}
\node[anchor=west,font=\fontsize{{{font_size + 1:g}}}{{{(font_size + 1) * 1.2:g}}}\selectfont\bfseries]
    at (0.15,1.3) {{{title}}};
"""]
    positions = {}
    for index, node in enumerate(nodes):
        row, col = divmod(index, columns)
        col = columns - 1 - col if row % 2 else col
        x, y = (col + 0.5) * width / columns, -row * row_pitch
        name, kind = node["name"], node["kind"]
        positions[name] = (row, x, y)
        lines = [rf"\textbf{{{node['title']}}}", node.get("shape", ""), node.get("detail", "")]
        label = r"\\[2pt]".join(line for line in lines if line)
        if kind == "operation":
            text_width = 1.35 if name == "concat" else 2.4
            layers.append(rf"""
\node[publication operation,text width={text_width}cm] ({name}) at ({x:.4f},{y:.4f}) {{{label}}};
\coordinate ({name}-west) at ({name}.west);
\coordinate ({name}-east) at ({name}.east);
\coordinate ({name}-north) at ({name}.north);
\coordinate ({name}-south) at ({name}.south);
""")
        else:
            box_width = 1.6 if kind in ("conv", "dense") else 0.8
            box_height = 4 if kind in ("conv", "pool") else 1.3
            band = (r"\ConvReluColor" if kind == "conv" else r"\FcReluColor") if node.get("band") else None
            layers.append(block(name, f"({x - box_width * 0.1:.4f},{y:.4f},0)",
                                width=box_width, height=box_height, depth=node.get("depth", 10),
                                color=palette[kind], band=band))
            layers.append(rf"\node[publication label] at ({x:.4f},{y - 1.0:.4f}) {{{label}}};" + "\n")
    for source, target in zip(nodes, nodes[1:]):
        a, b = source["name"], target["name"]
        row_a, x_a, y_a = positions[a]
        row_b, x_b, y_b = positions[b]
        if row_a == row_b:
            a_side, b_side = ("east", "west") if x_b > x_a else ("west", "east")
            layers.append(rf"\draw[publication flow] ({a}-{a_side}) -- ({b}-{b_side});" + "\n")
        else:
            side = "east" if row_a % 2 == 0 else "west"
            route_x = width - 0.07 if side == "east" else 0.07
            layers.append(rf"\draw[publication flow] ({a}-{side}) -- ({route_x:.4f},{y_a:.4f}) -- ({route_x:.4f},{y_b:.4f}) -- ({b}-{side});" + "\n")
    if auxiliary:
        target = auxiliary["target"]
        _, x, y = positions[target]
        layers.append(rf"""
\node[publication operation,fill=DataColor!15,text width=2.0cm,inner sep=2pt]
    (auxiliary) at ({x:.4f},{y + 1.15:.4f}) {{{auxiliary['text']}}};
\draw[publication flow] (auxiliary.south) -- ({target}-north);
""")
    # Fix the physical width so \includegraphics[width=\columnwidth] only scales
    # when the journal's column actually differs from the selected preset.
    layers.append(rf"""
\pgfresetboundingbox
\path[use as bounding box] (0,{bottom:.4f}) rectangle ({width},1.65);
""")
    layers.append(to_end())
    return layers


def copy_layer_assets(output_dir):
    """Package genuine library assets without restoring deleted submodule files."""
    destination = output_dir / "plotneuralnet"
    sources = {}
    for relative in ("layers/init.tex", "layers/Box.sty", "layers/RightBandedBox.sty", "layers/Ball.sty", "LICENSE"):
        local = PLOT_NEURAL_NET_ROOT / relative
        if local.is_file():
            content = local.read_bytes()
            origin = "working tree"
        else:
            result = subprocess.run(["git", "-C", str(PLOT_NEURAL_NET_ROOT), "show", f"HEAD:{relative}"], capture_output=True)
            if result.returncode:
                raise FileNotFoundError(f"PlotNeuralNet asset {relative} is missing from both the working tree and local Git history.")
            content = result.stdout
            origin = "local Git HEAD"
        target = destination / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(content)
        sources[relative] = dict(origin=origin, sha256=hashlib.sha256(content).hexdigest())
    return sources


def export_diagram(architecture, metadata, output_pdf, *, tex_only=False,
                   figure_caption=None, layout="single-column"):
    """Write TikZ, model metadata and library assets, then optionally compile PDF."""
    output_pdf = Path(output_pdf).expanduser().resolve()
    output_pdf.parent.mkdir(parents=True, exist_ok=True)
    metadata["plotneuralnet_assets"] = copy_layer_assets(output_pdf.parent)
    output_tex = output_pdf.with_suffix(".tex")
    with redirect_stdout(io.StringIO()):
        to_generate(architecture, str(output_tex))
    output_pdf.with_suffix(".json").write_text(json.dumps(metadata, indent=2) + "\n")
    if figure_caption is not None:
        caption_path = output_pdf.with_name(output_pdf.stem + "_caption.tex")
        caption_path.write_text(figure_caption.strip() + "\n")
        figure_path = output_pdf.with_name(output_pdf.stem + "_figure.tex")
        environment = "figure*" if layout == "double-column" else "figure"
        target_width = r"\textwidth" if layout == "double-column" else r"\columnwidth"
        # Relative paths work when the snippet is copied next to the PDF/caption.
        figure_path.write_text(
            "% Requires \\usepackage{graphicx}; keep this snippet, PDF and caption together.\n"
            "% From another directory use \\usepackage{import} and \\subimport{path/}{" + figure_path.name + "}.\n"
            "% Labels are 9 pt by default at the native 85 mm / 178 mm width.\n"
            + r"\begin{" + environment + "}[t]\n"
            + r"  \centering" + "\n"
            + r"  \includegraphics[width=" + target_width + "]{" + output_pdf.name + "}\n"
            + r"  \caption{" + "\n    " + r"\input{" + caption_path.name + "}\n  }\n"
            + r"  \label{fig:" + output_pdf.stem.replace("_", "-") + "}\n"
            + r"\end{" + environment + "}\n"
        )
        print(f"Editable LaTeX caption and figure snippet written to: {caption_path}, {figure_path}")
    print(f"TikZ source and verified tensor shapes written to: {output_tex}")
    if tex_only:
        return
    pdflatex = shutil.which("pdflatex")
    if not pdflatex:
        raise RuntimeError("pdflatex was not found. Use --tex-only or install a LaTeX distribution.")
    with tempfile.TemporaryDirectory(prefix="plotneuralnet-diagram-") as temp_dir:
        result = subprocess.run([pdflatex, "-interaction=nonstopmode", "-halt-on-error", "-no-shell-escape",
                                 "-output-directory", temp_dir, output_tex.name], cwd=output_tex.parent,
                                capture_output=True, text=True)
        if result.returncode:
            raise RuntimeError(f"pdflatex failed:\n{result.stdout[-8000:]}\n{result.stderr}")
        shutil.copy2(Path(temp_dir) / (output_tex.stem + ".pdf"), output_pdf)
    print(f"PDF diagram written to: {output_pdf}")
