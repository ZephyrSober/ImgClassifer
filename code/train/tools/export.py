from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np

TRAIN_SRC = Path(__file__).resolve().parents[1] / "src"
if str(TRAIN_SRC) not in sys.path:
    sys.path.insert(0, str(TRAIN_SRC))

from mindspore import Tensor, context, export, load_checkpoint, load_param_into_net, nn  # noqa: E402

from models.shuffleNet import build_model  # noqa: E402


class _SoftmaxNet(nn.Cell):

    def __init__(self, backbone: nn.Cell):
        super().__init__(auto_prefix=False)
        self.backbone = backbone
        self.softmax = nn.Softmax(axis=1)

    def construct(self, x: Tensor) -> Tensor:
        return self.softmax(self.backbone(x))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export a trained classifier to MindIR for MindSpore Lite deployment."
    )
    parser.add_argument("--ckpt", type=Path, required=True, help="Path to the trained .ckpt checkpoint.")
    parser.add_argument(
        "--model",
        default="shufflenet_v2_x1_0",
        help="Model name: shufflenet_v2_x1_0 / simple_cnn / linear_classifier.",
    )
    parser.add_argument("--output", type=str, default="shufflenetv2", help="Output file name without extension.")
    parser.add_argument("--num-classes", type=int, default=2, help="Number of output classes.")
    parser.add_argument("--image-size", type=int, default=224, help="Square input size (H == W).")
    parser.add_argument("--device", type=str, default="CPU", help="Device target used for export.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    if not args.ckpt.exists():
        raise FileNotFoundError(f"Checkpoint does not exist: {args.ckpt}")

    context.set_context(mode=context.GRAPH_MODE, device_target=args.device)

    model_cfg = {
        "model_name": args.model,
        "num_classes": args.num_classes,
        "pretrained": False,
        "image_size": args.image_size,
    }
    logits_net = build_model(model_cfg)  
    net = _SoftmaxNet(logits_net)      

    params = load_checkpoint(str(args.ckpt))
    load_param_into_net(net, params, strict_load=False)
    net.set_train(False)

    input_array = np.random.uniform(0.0, 1.0, size=[1, 3, args.image_size, args.image_size]).astype(np.float32)
    output_path = os.path.abspath(f"{args.output}.mindir")
    export(net, Tensor(input_array), file_name=output_path, file_format="MINDIR")

    print(f"Exported MindIR to: {output_path}")
    print(
        "Next step: converter_lite --fmk=MINDIR --modelFile=%s --outputFile=pet" % output_path
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
