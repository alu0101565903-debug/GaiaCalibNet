#!/usr/bin/env python3
"""Draw DirectConvPINN using PlotNeuralNet's VGG16 visual conventions.

The default hidden width comes from DirectConvPINN (128). Publication output
includes a vector PDF, editable LaTeX caption, figure inclusion snippet, verified
tensor shapes and original PlotNeuralNet layer assets. The compact default fits
one column; --layout double-column provides a wider arrangement.
"""
from __future__ import annotations

import argparse
import hashlib
import inspect
from pathlib import Path

import numpy as np
import torch
from torch import nn

if __package__:
    from .plotneuralnet_utils import (
        REPO_ROOT, add_publication_args, export_diagram, publication_architecture,
    )
else:
    from src.utils.diagram_pinn_architecture.plotneuralnet_utils import (
        REPO_ROOT, add_publication_args, export_diagram, publication_architecture,
    )

from src.models.pinn_architecture import DirectConvPINN

MODEL_DEFAULTS = inspect.signature(DirectConvPINN).parameters


def _inspect_model(input_dim, num_params, num_components, hidden_dim):
    """Run the real forward/reconstruct methods; fail if the renderer is outdated."""
    if input_dim < 1 or num_params < 0 or num_components < 1 or hidden_dim < 2:
        raise ValueError("input_dim and num_components must be positive; num_params >= 0; hidden_dim >= 2")
    with torch.random.fork_rng():
        model = DirectConvPINN(input_dim, np.zeros((num_components, input_dim)),
                               np.zeros(input_dim), num_params=num_params, hidden_dim=hidden_dim).eval()
    extractor = list(model.conv_feature_extractor)
    head = list(model.mlp)
    if ([type(layer) for layer in extractor] != [nn.Conv1d, nn.GELU, nn.Conv1d, nn.GELU, nn.AdaptiveAvgPool1d]
            or [type(layer) for layer in head] != [nn.Linear, nn.GELU, nn.Linear, nn.GELU, nn.Linear]):
        raise ValueError("DirectConvPINN layers have changed; update the diagram layout before exporting.")
    observed = {}

    def capture(name):
        def hook(module, inputs, output):
            observed[name] = dict(input=list(inputs[0].shape), output=list(output.shape))
        return hook

    names = ("conv1", "gelu1", "conv2", "gelu2", "pool", "fc1", "gelu3", "fc2", "gelu4", "fc3")
    hooks = [layer.register_forward_hook(capture(name)) for name, layer in zip(names, [*extractor, *head])]
    try:
        with torch.no_grad():
            x = torch.ones((2, input_dim))
            params = torch.zeros((2, num_params)) if num_params else None
            coefficients = model(x, params)
            ratio = model.reconstruct(coefficients)
    finally:
        for hook in hooks:
            hook.remove()
    assert list(coefficients.shape) == [2, num_components]
    assert list(ratio.shape) == [2, input_dim]
    assert torch.count_nonzero(coefficients).item() == 0
    convolutions = []
    for layer in (extractor[0], extractor[2]):
        convolutions.append(dict(in_channels=layer.in_channels, out_channels=layer.out_channels,
                                 kernel=layer.kernel_size[0], stride=layer.stride[0],
                                 padding=layer.padding[0], padding_mode=layer.padding_mode))
    return dict(input_dim=input_dim, num_params=num_params, num_components=num_components,
                hidden_dim=hidden_dim, hidden_half=head[2].out_features,
                flat_features=observed["pool"]["output"][1] * observed["pool"]["output"][2],
                combined_features=head[0].in_features, convolutions=convolutions, observed=observed,
                trainable_parameters=sum(p.numel() for p in model.parameters()),
                model_sha256=hashlib.sha256(Path(inspect.getfile(DirectConvPINN)).read_bytes()).hexdigest())


def _architecture(input_dim, num_params, num_components, hidden_dim, *, metadata=None,
                  layout="single-column", font_size=9):
    """Render the verified model with compact, uniformly readable stage labels."""
    spec = metadata or _inspect_model(input_dim, num_params, num_components, hidden_dim)
    conv1, conv2 = spec["convolutions"]
    shapes = spec["observed"]
    length1, length2 = shapes["conv1"]["output"][-1], shapes["conv2"]["output"][-1]
    channels1, channels2 = conv1["out_channels"], conv2["out_channels"]
    pooled_length = shapes["pool"]["output"][-1]
    flat, combined, half = spec["flat_features"], spec["combined_features"], spec["hidden_half"]
    longest_spectrum = max(input_dim, length1, length2, pooled_length)

    def spectral_depth(length):
        # Preserve the 86 -> 128 pooling increase, with a legible minimum depth.
        return max(4, 12 * length / longest_spectrum)

    def dense_depth(features):
        return max(4, 12 * features / max(hidden_dim, half, num_components))

    nodes = [dict(name="input", title="Normalized input",
                  shape=rf"$1\times {input_dim}$", kind="data",
                  depth=spectral_depth(input_dim))]
    for name, channels, length in (
        ("conv1", channels1, length1),
        ("conv2", channels2, length2),
    ):
        nodes.append(dict(name=name, title="Conv1d + GELU", kind="conv", band=True,
                          shape=rf"${channels}\times {length}$",
                          depth=spectral_depth(length)))
    nodes.extend([
        dict(name="pool", title="Adaptive avg. pool", kind="pool",
             shape=rf"${channels2}\times {pooled_length}$",
             depth=spectral_depth(pooled_length)),
        dict(name="flatten", title="Flatten", kind="data", shape=rf"${flat}$", depth=12),
    ])
    auxiliary = None
    if num_params:
        nodes.append(dict(name="concat", title="Concat", kind="operation", shape=rf"${combined}$"))
        auxiliary = dict(target="concat", text=rf"$p\in\mathbb{{R}}^{{{num_params}}}$")
    for name, features, activated in (
        ("fc1", hidden_dim, True),
        ("fc2", half, True),
        ("fc3", num_components, False),
    ):
        nodes.append(dict(name=name, title="Linear + GELU" if activated else "Linear",
                          shape=rf"${features}$" if activated else rf"$c\in\mathbb{{R}}^{{{features}}}$",
                          kind="dense", band=activated, depth=dense_depth(features)))
    nodes.extend([
        dict(name="reconstruct", title="Fixed PCA", kind="operation",
             shape=r"$R=\exp(\mu+cV)$"),
        dict(name="ratio", title="Calibration ratio", kind="data",
             shape=rf"$R\in\mathbb{{R}}_{{>0}}^{{{input_dim}}}$",
             depth=spectral_depth(input_dim)),
    ])
    return publication_architecture("DirectConvPINN", nodes, layout=layout,
                                    font_size=font_size, auxiliary=auxiliary)


def _figure_caption(spec):
    """Keep publication prose separate from the vector drawing for easy editing."""
    conv1, conv2 = spec["convolutions"]
    input_dim, num_params = spec["input_dim"], spec["num_params"]
    components, hidden = spec["num_components"], spec["hidden_dim"]
    pooled_length = spec["observed"]["pool"]["output"][-1]
    input_pool_length = spec["observed"]["conv2"]["output"][-1]
    chunks = [
        r"Architecture of DirectConvPINN. Arrows give the order of operations; "
        r"tensor labels omit the batch axis $B$ and show channels $\times$ "
        r"wavelength bins for feature maps, or feature counts for vectors. ",
        rf"The illustrated configuration has $N={input_dim}$ spectral bins, "
        rf"$P={num_params}$ auxiliary inputs, $K={components}$ PCA components "
        rf"and hidden width $H={hidden}$. ",
        r"For each input spectrum $x\in\mathbb{R}^{B\times N}$, "
        r"$s=\max_{\lambda}|x|+10^{-20}$ and "
        r"$x_{\mathrm{norm}}=\operatorname{unsqueeze}(x/s)$ adds the channel axis. ",
        rf"The two Conv1d layers use $(k,s,p)=({conv1['kernel']},{conv1['stride']},{conv1['padding']})$ "
        rf"and $({conv2['kernel']},{conv2['stride']},{conv2['padding']})$, respectively, "
        r"where $k$, $s$ and $p$ denote kernel size, stride and padding. ",
        rf"Adaptive average pooling maps {input_pool_length} to {pooled_length} spectral bins "
        rf"and flattening produces {spec['flat_features']} features. ",
    ]
    if num_params:
        auxiliary_names = (r"BP$-$RP, G and RUWE" if num_params == 3
                           else r"BP$-$RP and G" if num_params == 2
                           else rf"the {num_params} supplied features")
        chunks.append(rf"The auxiliary inputs $p$ ({auxiliary_names}), standardized before entering "
                      rf"the model, are concatenated to give {spec['combined_features']} features. ")
    chunks.extend([
        rf"The dense head has widths {hidden}, {spec['hidden_half']} and {components} and "
        r"its final linear layer predicts the PCA coefficients $c$. ",
        r"Yellow blocks denote convolutions, red blocks pooling, blue blocks linear layers, "
        r"and shaded bands denote GELU activations; muted blue blocks show data reshaping or tensors. "
        r"Block depths reflect spectral lengths or feature counts, with a minimum size "
        r"for readability; other block dimensions are schematic. ",
        r"The non-trainable reconstruction uses $R=\exp(\mu+cV)$, with "
        r"$V\in\mathbb{R}^{K\times N}$ and $\mu\in\mathbb{R}^{N}$ stored as fixed buffers "
        r"from PCA fitted on training-set logarithmic calibration ratios. ",
        r"The output layer is initialized with $W=0$ and $b=0$, so initially "
        r"$c=0$ and $R=\exp(\mu)$. Calibrated flux is computed outside the model "
        r"as $\widehat{Y}=x\odot R$.",
    ])
    return "".join(chunks)


def _parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dim", type=int, default=343)
    parser.add_argument("--num-params", type=int, default=MODEL_DEFAULTS["num_params"].default)
    parser.add_argument("--num-components", type=int, default=20)
    parser.add_argument("--hidden-dim", type=int, default=MODEL_DEFAULTS["hidden_dim"].default,
                        help="Hidden width (default: DirectConvPINN's constructor default).")
    parser.add_argument("--output", type=Path, default=REPO_ROOT / "results/diagrams/directconvpinn_architecture.pdf")
    parser.add_argument("--tex-only", action="store_true", help="Write source and assets without compiling the PDF.")
    add_publication_args(parser)
    args = parser.parse_args()
    if args.input_dim < 1 or args.num_params < 0 or args.num_components < 1 or args.hidden_dim < 2:
        parser.error("input-dim/components must be positive; num-params >= 0; hidden-dim >= 2")
    if args.output.suffix.lower() != ".pdf":
        parser.error("--output must have a .pdf extension")
    return args


def main():
    args = _parse_args()
    metadata = _inspect_model(args.input_dim, args.num_params, args.num_components, args.hidden_dim)
    metadata["publication"] = dict(layout=args.layout, font_size_pt=args.font_size,
                                   width_mm=85 if args.layout == "single-column" else 178)
    architecture = _architecture(args.input_dim, args.num_params, args.num_components, args.hidden_dim,
                                 metadata=metadata, layout=args.layout, font_size=args.font_size)
    export_diagram(architecture, metadata, args.output, tex_only=args.tex_only,
                   figure_caption=_figure_caption(metadata), layout=args.layout)


if __name__ == "__main__":
    main()
