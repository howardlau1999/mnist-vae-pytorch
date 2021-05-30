import os
import torch
import random
import torchvision.transforms
from torch.utils.data import sampler
from torch.utils.data.dataloader import DataLoader, RandomSampler
from torchvision.datasets import MNIST, FashionMNIST
from torchvision.utils import make_grid, save_image
import logging
import matplotlib.pyplot as plt
from argparse import ArgumentParser
from tqdm import tqdm
import torch.nn.functional as F
import torch.optim
import numpy as np

from decoder import ConvDecoder, MLPDecoder
from encoder import ConvEncoder, MLPEncoder

from cvae import VAELoss, VAEParameters, MNISTCVAE
from cvqvae import VQVAELoss, MNISTCVQVAE, CVQVAECodebook, VQVAEPerplexity

logger = logging.getLogger()
logging.basicConfig(format="%(asctime)s %(levelname)s %(message)s", level=logging.INFO)

DATASETS = {
    "mnist": MNIST,
    "fashionmnist": FashionMNIST,
}

ENCODERS = {
    "mlp": MLPEncoder,
    "conv": ConvEncoder,
}

DECODERS = {
    "mlp": MLPDecoder,
    "conv": ConvDecoder,
}

PARAMS = {
    "mlp": {"embed_dim": 256},
    "conv": {"embed_dim": 512},
}

if __name__ == "__main__":    
    parser = ArgumentParser()

    parser.add_argument("--batch_size", type=int, default=2048)
    parser.add_argument("--num_epochs", type=int, default=100)
    parser.add_argument("--learning_rate", type=float, default=1e-3)
    parser.add_argument("--no_cuda", action="store_true")
    parser.add_argument("--latent_dim", type=int, default=3)
    parser.add_argument("--encoder", type=str, choices=ENCODERS.keys(), default="mlp")
    parser.add_argument("--decoder", type=str, choices=DECODERS.keys(), default="mlp")
    parser.add_argument("--dataset", type=str, choices=DATASETS.keys(), default="mnist")
    parser.add_argument("--image_dir", type=str, default="images")
    parser.add_argument("--seed", type=int, default=315)
    parser.add_argument("--export_onnx", action="store_true")
    parser.add_argument("--model_path", type=str, default=None)
    parser.add_argument("--model_type", type=str, choices=["cvqvae", "cvae"], default="cvae")

    args = parser.parse_args()
    logger.info(f"Training args: {args}")

    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available() and not args.no_cuda:
        torch.cuda.manual_seed_all(args.seed)

    device = "cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu"
    encoder = ENCODERS[args.encoder]()
    decoder = DECODERS[args.decoder](latent_dim=args.latent_dim)
    if args.model_type == "cvae":
        param = VAEParameters(**{"latent_dim": args.latent_dim, **PARAMS[args.encoder]})
        model = MNISTCVAE(encoder, decoder, param)
    else:
        perplexity_fn = VQVAEPerplexity().to(device)
        codebook = CVQVAECodebook(latent_dim=args.latent_dim)
        model = MNISTCVQVAE(encoder, decoder, codebook, **{"latent_dim": args.latent_dim, **PARAMS[args.encoder]})
    logger.info(f"Model initialized.")

    if args.export_onnx:
        from export_onnx import save_decoder_onnx
        save_decoder_onnx(args, model)

    model = model.to(device)
    train_dataset = DATASETS[args.dataset]("./", download=True, transform=torchvision.transforms.ToTensor())
    logger.info(f"Loaded {args.dataset} train dataset")

    sampler = RandomSampler(train_dataset)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, sampler=sampler)

    loss_fn = VAELoss() if args.model_type == "cvae" else VQVAELoss()
    loss_fn = loss_fn.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)

    model.zero_grad()
    eval_idx = torch.randint(0, model.codebook.num_embeddings, (100, ), dtype=torch.long).to(device)
    eval_z = torch.randn((100, args.latent_dim)).to(device)
    eval_labels = torch.zeros((100, ), dtype=torch.long).to(device)
    for i in range(100):
        eval_labels[i] = i // 10
    eval_labels = F.one_hot(eval_labels, num_classes=10)
    def evaluate(show=False, epoch=0):
        model.eval()
        with torch.no_grad():
            if args.model_type == "cvae":
                reconstruction = model.decoder(eval_z, eval_labels).reshape(100, 1, 28, 28)
            else:
                reconstruction = model.decoder(model.codebook.embedding(torch.randint(0, model.codebook.num_embeddings, (100, ), dtype=torch.long).to(device)), eval_labels).reshape(100, 1, 28, 28)
            if show:
                grid = make_grid(reconstruction.detach().cpu(), nrow=10).numpy()
                plt.imshow(grid.transpose(1, 2, 0), cmap="gray")
                plt.show()
            else:
                if not os.path.exists(args.image_dir):
                    os.makedirs(args.image_dir)
                save_image(reconstruction, f"{args.image_dir}/eval_{epoch}.png", nrow=10)

    for epoch in tqdm(range(args.num_epochs)):
        model.train()
        tr_loss = 0
        tr_step = 0
        with tqdm(train_loader) as t:
            for batch in t:
                images, labels = (x.to(device) for x in batch)
                labels = F.one_hot(labels, num_classes=10)
                outputs = model(images, labels)
                loss = loss_fn(images, *outputs)
                loss.backward()
                optimizer.step()
                model.zero_grad()

                tr_loss += loss.item()
                tr_step += 1
                
                status_dict = {"loss": tr_loss / tr_step}
                if args.model_type == "cvqvae":
                    perplexity = perplexity_fn(outputs[-1])
                    status_dict.update({"perplextiy": perplexity.item()})
                t.set_postfix(status_dict)
        evaluate(epoch=epoch)
    
    logger.info("Saving model to model.pth")
    torch.save(model.state_dict(), "model.pth")
    torch.save(optimizer.state_dict(), "optimizer.pth")
    logger.info("Saved")