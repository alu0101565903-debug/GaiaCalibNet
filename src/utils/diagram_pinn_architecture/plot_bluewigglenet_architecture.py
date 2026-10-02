#!/usr/bin/env python3
"""Draw BlueWiggleNet using the same PlotNeuralNet style as DirectConvPINN.

The default input has 57 blue bins (336 <= wavelength < 450 nm, step 2 nm).
The PDF, TikZ source, verified shapes and PlotNeuralNet assets are saved together.
"""
from __future__ import annotations

import argparse
import hashlib
import inspect
from pathlib import Path

import torch
from torch import nn

if __package__:
    from .plotneuralnet_utils import (
        REPO_ROOT, block, caption, export_diagram, operation,
        to_begin, to_connection, to_cor, to_end, to_head,
    )
else:
    from plotneuralnet_utils import (
        REPO_ROOT, block, caption, export_diagram, operation,
        to_begin, to_connection, to_cor, to_end, to_head,
    )

from src.models.pinn_architecture import BlueWiggleNet

MODEL_DEFAULTS = inspect.signature(BlueWiggleNet).parameters


def _inspect_model(input_dim, num_params, hidden_dim):
    """Read layer settings and verify tensor shapes with the actual forward pass."""
    if input_dim < 1 or num_params < 0 or hidden_dim < 1:
        raise ValueError("input_dim and hidden_dim must be positive; num_params >= 0")
    with torch.random.fork_rng():
        model = BlueWiggleNet(input_dim, num_params=num_params, hidden_dim=hidden_dim).eval()
    extractor, head = list(model.conv), list(model.mlp)
    if ([type(layer) for layer in extractor] != [nn.Conv1d, nn.GELU, nn.Conv1d, nn.GELU, nn.Flatten]
            or [type(layer) for layer in head] != [nn.Linear, nn.GELU, nn.Dropout, nn.Linear]):
        raise ValueError("BlueWiggleNet layers have changed; update the diagram before exporting.")
    observed = {}
    captured = {}

    def capture(name):
        def hook(module, inputs, output):
            observed[name] = dict(input=list(inputs[0].shape), output=list(output.shape))
            if name == "conv1":
                captured["normalized_input"] = inputs[0].detach().clone()
            elif name == "fc2":
                captured["log_ratio"] = output.detach().clone()
        return hook

    names = ("conv1", "gelu1", "conv2", "gelu2", "flatten", "fc1", "gelu3", "dropout", "fc2")
    hooks = [layer.register_forward_hook(capture(name)) for name, layer in zip(names, [*extractor, *head])]
    try:
        with torch.no_grad():
            x = torch.linspace(-2e-16, 3e-16, input_dim).repeat(2, 1)
            params = torch.zeros((2, num_params)) if num_params else None
            ratio = model(x, params)
            scale = x.abs().amax(dim=1, keepdim=True) + 1e-20
            torch.testing.assert_close(captured["normalized_input"], (x / scale).unsqueeze(1))
            torch.testing.assert_close(ratio, captured["log_ratio"].exp())
            torch.testing.assert_close(ratio, torch.ones_like(x))
    finally:
        for hook in hooks:
            hook.remove()
    if list(ratio.shape) != [2, input_dim]:
        raise ValueError("BlueWiggleNet no longer produces one ratio per input bin.")
    convolutions = [dict(in_channels=layer.in_channels, out_channels=layer.out_channels,
                         kernel=layer.kernel_size[0], stride=layer.stride[0],
                         padding=layer.padding[0], padding_mode=layer.padding_mode)
                    for layer in (extractor[0], extractor[2])]
    return dict(model="BlueWiggleNet", input_dim=input_dim, num_params=num_params,
                hidden_dim=hidden_dim, dropout=head[2].p,
                flat_features=observed["flatten"]["output"][-1],
                combined_features=head[0].in_features, convolutions=convolutions,
                observed=observed, output_shape=list(ratio.shape), initial_ratio=1.0,
                trainable_parameters=sum(p.numel() for p in model.parameters()),
                model_sha256=hashlib.sha256(Path(inspect.getfile(BlueWiggleNet)).read_bytes()).hexdigest())


def _architecture(input_dim, num_params, hidden_dim, *, metadata=None):
    spec = metadata or _inspect_model(input_dim, num_params, hidden_dim)
    conv1, conv2 = spec["convolutions"]
    length1, length2 = (spec["observed"][name]["output"][-1] for name in ("conv1", "conv2"))
    channels1, channels2 = conv1["out_channels"], conv2["out_channels"]
    flat, combined, dropout = spec["flat_features"], spec["combined_features"], spec["dropout"]

    def spectral_depth(length):
        return min(40, max(8, 32 * (length / input_dim) ** 0.5))

    output_depth = min(38, max(10, 30 * (input_dim / hidden_dim) ** 0.5))
    layers = [to_head("plotneuralnet"), r"""
\usepackage{amsmath,amssymb}
\usetikzlibrary{calc,arrows.meta}
""", to_cor(), to_begin(), r"""
% Same translucent boxes, GELU bands, colors and arrows as DirectConvPINN / VGG16.
\definecolor{DataColor}{HTML}{9DB6C4}
\definecolor{DropoutColor}{HTML}{75B5A1}
\tikzset{operation/.style={draw=black!55,rounded corners=2pt,align=center,
    inner sep=7pt,font=\normalsize},
    note/.style={align=left,font=\small,text=black!75}}
\coordinate (caption-row) at (0,-3.15);
\node[anchor=west,font=\Large\bfseries] at (-1.2,5.7) {BlueWiggleNet};
\node[anchor=west,font=\bfseries,text=black!65] at (-1.2,4.7)
    {\texttt{forward(x\_blue, p)}: pixel-wise multiplicative correction};
""",
        rf"\node[anchor=west,text=black!65] at (5.0,5.7) {{$N_b={input_dim}$ blue wavelength bins\quad $P={num_params}$ auxiliary inputs\quad $H={hidden_dim}$}};" + "\n",
        block("input", "(0,0,0)", width=1.5, height=3, depth=spectral_depth(input_dim),
              channels=1, length=input_dim, color="DataColor"),
        caption("input", "Normalized input", r"$x_{\rm norm}$"),
    ]
    for name, conv, channels, length, x in (
        ("conv1", conv1, channels1, length1, 3.6),
        ("conv2", conv2, channels2, length2, 7.8),
    ):
        layers.extend([
            block(name, f"({x},0,0)", width=3, height=14, depth=spectral_depth(length),
                  channels=channels, length=length, band=r"\ConvReluColor"),
            caption(name, r"Conv1d\\+ GELU", rf"$k={conv['kernel']},\ s={conv['stride']},\ p={conv['padding']}$"),
        ])
    layers.extend([
        block("flatten", "(12.0,0,0)", width=2, height=3, depth=46,
              length=flat, color="DataColor"),
        caption("flatten", "Flatten", rf"${channels2}\times {length2}={flat}$"),
    ])
    if num_params:
        layers.extend([
            r"\node[circle,draw=black!60,fill=white,minimum size=0.7cm,inner sep=0pt] (concat) at (16.2,0) {$\Vert$};" + "\n",
            r"\coordinate (concat-west) at (concat.west);\coordinate (concat-east) at (concat.east);\coordinate (concat-anchor) at (concat.center);" + "\n",
            caption("concat", "Concatenate", rf"${flat}+{num_params}={combined}$"),
            operation("params", rf"\textbf{{Auxiliary inputs}}\\$p\in\mathbb{{R}}^{{B\times {num_params}}}$", 16.2, 3.1, width=2.7, fill="DataColor!15"),
            r"\draw[connection] (params-south) -- node {\midarrow} (concat.north);" + "\n",
        ])
    layers.extend([
        block("fc1", "(19.1,0,0)", width=2.5, height=3, depth=30,
              length=hidden_dim, color=r"\FcColor", band=r"\FcReluColor"),
        caption("fc1", r"Linear\\+ GELU", rf"${combined}\to {hidden_dim}$"),
        block("dropout", "(22.8,0,0)", width=1.1, height=3, depth=30,
              length=hidden_dim, color="DropoutColor"),
        caption("dropout", "Dropout", rf"$p={dropout:g}$\\training only"),
        block("fc2", "(26.0,0,0)", width=2.5, height=3, depth=output_depth,
              length=input_dim, color=r"\FcColor"),
        caption("fc2", "Linear", rf"${hidden_dim}\to {input_dim}$"),
        operation("exp", r"\textbf{Exponential}\\[5pt]$R_{\rm blue}=\exp(z)$", 30.5, 0, width=2.6, fill="gray!5"),
        caption("exp", "Element-wise", "positive ratios"),
        block("ratio", "(35.0,0,0)", width=1.5, height=3, depth=spectral_depth(input_dim),
              length=input_dim, color="DataColor"),
        caption("ratio", "Blue correction", rf"$R_{{\rm blue}}\in\mathbb{{R}}_{{>0}}^{{B\times {input_dim}}}$"),
    ])
    chain = ["input", "conv1", "conv2", "flatten"]
    if num_params:
        chain.append("concat")
    chain.extend(["fc1", "dropout", "fc2", "exp", "ratio"])
    layers.extend(to_connection(a, b) for a, b in zip(chain, chain[1:]))
    layers.extend([
        rf"\node[above=6pt,align=center,font=\small] at ($(fc2-east)!0.5!(exp-west)$) {{$z$\\$(B,{input_dim})$}};" + "\n",
        r"""
\draw[densely dashed,black!45]
    (conv2-nearnortheast) -- (flatten-nearnorthwest)
    (conv2-nearsoutheast) -- (flatten-nearsouthwest)
    (conv2-farnortheast) -- (flatten-farnorthwest)
    (conv2-farsoutheast) -- (flatten-farsouthwest);
\node[note,anchor=north west] at (-1.2,-5.0)
    {\textbf{Input normalization} (original Gaia blue flux)\\[4pt]
     $s=\max_\lambda |x_{\rm blue}|+10^{-20}$,\quad
     $x_{\rm norm}=\operatorname{unsqueeze}(x_{\rm blue}/s)$};
\node[note,anchor=north west] at (10.0,-5.0)
    {\textbf{Tensor labels} (batch axis $B$ omitted)\\[4pt]
     Feature maps: channels and blue wavelength bins; vectors: features.\\
     Convolutions preserve wavelength sampling; block sizes are schematic.};
\node[note,anchor=north west] at (24.0,-5.0)
    {\textbf{Output semantics}\\[4pt]
     Final linear layer: $W=0$, $b=0$ initially.\\
     Hence $z=0$ and $R_{\rm blue}=1$: the initial correction is neutral.};
\node[note,anchor=west] at (-1.2,-7.0) {\textbf{Legend}};
\path[draw,fill=\ConvColor,fill opacity=0.6] (0.7,-7.12) rectangle ++(0.45,0.25);
\node[note,anchor=west] at (1.3,-7.0) {Conv1d};
\path[draw,fill=\FcColor,fill opacity=0.6] (4.1,-7.12) rectangle ++(0.45,0.25);
\node[note,anchor=west] at (4.7,-7.0) {Linear};
\path[draw,fill=DropoutColor,fill opacity=0.6] (7.3,-7.12) rectangle ++(0.45,0.25);
\node[note,anchor=west] at (7.9,-7.0) {Dropout};
\path[draw,fill=\ConvReluColor,fill opacity=0.8] (11.0,-7.12) rectangle ++(0.22,0.25);
\path[draw,fill=\FcReluColor,fill opacity=0.8] (11.24,-7.12) rectangle ++(0.22,0.25);
\node[note,anchor=west] at (11.6,-7.0) {Bands: GELU};
""",
    ])
    if num_params:
        names = (r"BP$-$RP, G, RUWE" if num_params == 3
                 else r"BP$-$RP, G" if num_params == 2 else rf"{num_params} supplied features")
        layers.append(rf"\node[note,anchor=west] at (24.0,-7.0) {{Auxiliary inputs: {names}; standardized before the model.}};" + "\n")
    layers.append(to_end())
    return layers


def _parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dim", type=int, default=57,
                        help="Blue wavelength bins (default: 57, for 336 to <450 nm at 2 nm spacing).")
    parser.add_argument("--num-params", type=int, default=MODEL_DEFAULTS["num_params"].default)
    parser.add_argument("--hidden-dim", type=int, default=MODEL_DEFAULTS["hidden_dim"].default)
    parser.add_argument("--output", type=Path, default=REPO_ROOT / "results/diagrams/bluewigglenet_architecture.pdf")
    parser.add_argument("--tex-only", action="store_true", help="Write source and assets without compiling the PDF.")
    args = parser.parse_args()
    if args.input_dim < 1 or args.num_params < 0 or args.hidden_dim < 1:
        parser.error("input-dim and hidden-dim must be positive; num-params >= 0")
    if args.output.suffix.lower() != ".pdf":
        parser.error("--output must have a .pdf extension")
    return args


def main():
    args = _parse_args()
    metadata = _inspect_model(args.input_dim, args.num_params, args.hidden_dim)
    architecture = _architecture(args.input_dim, args.num_params, args.hidden_dim, metadata=metadata)
    export_diagram(architecture, metadata, args.output, tex_only=args.tex_only)


if __name__ == "__main__":
    main()
