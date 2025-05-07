import argparse
import h5py
import torch
import pickle
import requests
import numpy as np

from PIL import Image
from src import utils, modules


@torch.no_grad()
def load_model(use_cpu, gpu, background_path):
    """
    Load the GeoLocModel, preprocessing pipeline, and background collection.
    """
    print("\n> loading model")
    model = modules.GeoLocModel(pretrained=True).eval()
    if not use_cpu:
        model = model.to(gpu)
    transform = utils.Preprocessing("inference", backbone="efficientnet")
    mus = model.cls_head.mus.t().cpu().numpy()
    cells_assignments = pickle.load(open("data/cells_assignments.pkl", "rb"))

    print("\n> loading background collection")
    with h5py.File(background_path, "r", driver=None) as hdf5_file:
        back_col_emb = hdf5_file["features"][:, :].T.astype(np.float32)
        back_col_cells = hdf5_file["labels"][:, :]
    print("Background collection features:", back_col_emb.T.shape)
    print("Background collection cells:", back_col_cells.shape)

    return model, transform, mus, cells_assignments, back_col_emb, back_col_cells


@torch.no_grad()
def run_evaluation(
    model, transform, mus, cells_assignments, back_col_emb, back_col_cells, args, image=None, image_url=None
):
    """
    Run the evaluation process using the loaded model, preprocessing pipeline, and background collection.
    """
    print("\n> loading image")
    if image is not None:
        query_image = Image.open(image).convert("RGB")
    elif image_url is not None:
        img = requests.get(image_url, stream=True).raw
        query_image = Image.open(img).convert("RGB")
    else:
        raise ValueError("Either image or image_url must be provided")

    query_tensor = transform(query_image)
    print("Image tensor shape:", query_tensor.numpy().shape)

    print("\n> run inference")
    if not args.use_cpu:
        query_tensor = query_tensor.to(args.gpu)
    prediction, cell_probs, embeddings = model(query_tensor.unsqueeze(0))
    embeddings = embeddings.cpu().numpy()
    cell_probs = cell_probs[0].cpu().numpy()
    max_cell = np.argmax(cell_probs)

    # Search within Cell scheme
    if max_cell in cells_assignments:
        idxs = np.array(list(cells_assignments[max_cell]))
        sims = np.dot(embeddings, back_col_emb[:, idxs])[0]

        NNs = np.argsort(-sims)[: args.top_k]
        sims = sims[NNs]
        candidates = back_col_cells[idxs[NNs]]
        pr = utils.spatial_clustering(candidates, sims, radius=args.eps, a=0)
    else:
        pr = prediction[0].cpu().numpy()

    conf = utils.prediction_density(max_cell, cell_probs, mus)

    print("Prediction (Lat,Lon): ({:.4f}, {:.4f})".format(*pr))
    print("Confidence: {:.1f}%".format(conf[args.conf_scale] * 100))

    return {"prediction": {"latitude": pr[0], "longitude": pr[1]}, "confidence": conf[args.conf_scale] * 100}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("-ip", "--image_path", type=str, default=None)
    parser.add_argument("-iu", "--image_url", type=str, default=None)
    parser.add_argument(
        "-b", "--background", type=str, default="./back_coll_features.hdf5"
    )
    parser.add_argument("-g", "--gpu", type=int, default=0)
    parser.add_argument("-k", "--top_k", type=int, default=10)
    parser.add_argument("-e", "--eps", type=float, default=1.0)
    parser.add_argument("-cpu", "--use_cpu", action="store_true")
    parser.add_argument("-cs", "--conf_scale", type=int, default=25)
    args = parser.parse_args()

    if args.image_path is None and args.image_url is None:
        raise Exception("Please provide an image path or URL as input")

    model, transform, mus, cells_assignments, back_col_emb, back_col_cells = load_model(
        args.use_cpu, args.gpu, args.background
    )
    run_evaluation(
        model,
        transform,
        mus,
        cells_assignments,
        back_col_emb,
        back_col_cells,
        args,
        image=args.image_path,
        image_url=args.image_url,
    )
