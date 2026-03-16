import argparse
import torch

from models.sample_model import get_sample_model, get_sample_input
from graph.graph_extractor import extract_graph
from optimizer.operator_fusion import fuse_conv_relu


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--print-graph",
        action="store_true",
        help="Extract and print the computation graph for the sample model.",
    )
    parser.add_argument(
        "--optimize",
        action="store_true",
        help="Run optimizer passes (currently: Conv2D+ReLU fusion).",
    )
    args = parser.parse_args()

    model = get_sample_model().eval()
    x = get_sample_input()

    if args.print_graph:
        gir = extract_graph(model, x)
        print("Graph BEFORE optimization")
        print(gir.pretty())

        if args.optimize:
            gir_opt = fuse_conv_relu(gir)
            print("\nGraph AFTER optimization")
            print(gir_opt.pretty())

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

