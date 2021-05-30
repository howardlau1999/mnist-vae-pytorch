import os
import logging
import torch
import sys

logger = logging.getLogger()
logging.basicConfig(format="%(asctime)s %(levelname)s %(message)s", level=logging.INFO)

def save_decoder_onnx(args, model):
    if not args.model_path or not os.path.exists(args.model_path):
        logger.error(f"Cannot find the model at {args.model_path}")
        sys.exit(1)
    model_state_dict = torch.load(args.model_path, map_location="cpu")
    model.load_state_dict(model_state_dict)
    logger.info(f"Loaded model {args.model_path}")

    torch.onnx.export(
        model.decoder,
        (torch.rand(10, args.latent_dim), torch.eye(10)),
        "decoder.onnx",
        export_params=True,
        do_constant_folding=True,
        input_names=["z", "c"],
        output_names=["reconstruction"],
        dynamic_axes={
            "z": {0: "batch_size"},
            "c": {0: "batch_size"},
            "reconstruction": {0: "batch_size"},
        }
    )

    logger.info("Exported decoder to decoder.onnx")