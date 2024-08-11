import os
import warnings

import hydra
import pandas as pd
import torch
from omegaconf import DictConfig
from torch.utils.data import DataLoader

from src.config import get_model
from src.data import MyDataset, load_data

from tqdm import tqdm
import numpy as np

warnings.filterwarnings("ignore")

CONFIG_NAME = "visualize_loss_config"

def visualize(dataset, model, grid_dim, batch_size, scale, device):
    object0, label0 = dataset[0]
    n_dim = len(object0)

    # object0 = torch.Tensor
    # from out import object0
    object1 = object0 + torch.randn(n_dim, 1)
    # from out import object1
    object2 = object0 + torch.randn(n_dim, 1)

    # object1, label1 = dataset[15]
    # object2, label2 = dataset[30]

    # print(label0, label1, label2)

    # ==== search setup ====
    dir1 = object1 - object0
    dir1 = dir1 / dir1.norm() * scale

    dir2 = object2 - object0
    dir2 = dir2 / dir2.norm() * scale
    # dir2 = dir2 / dir2.norm() * dir1.norm()
    
    print('LABEL:', label0)
    print('MODEL OUTPUT:', model(object0[None, ...].to(device)))
    print('MODEL ATTACKED OUTPUT:', model(object1[None, ...].to(device)))
    print('DIR NORM:', dir1.norm(), dir2.norm())

    dir1 *= scale
    dir2 *= scale

    results = torch.zeros((grid_dim, grid_dim))
    coords1 = torch.linspace(-1, 1, grid_dim)
    coords2 = torch.linspace(-1, 1, grid_dim)

    batch = []
    batch_coords = []

    for i1 in tqdm(range(grid_dim)):
        for i2 in range(grid_dim):
            new_object = object0 + coords1[i1] * dir1 + coords2[i2] * dir2

            batch_coords.append((i1, i2))
            batch.append(new_object)

            if len(batch) >= batch_size:
                with torch.no_grad():
                    out = model(torch.stack(batch).to(device))
                j = 0
                for j1, j2 in batch_coords:
                    results[j1, j2] = out[j]
                    j += 1
                batch_coords = []
                batch = []

    with torch.no_grad():
        out = model(torch.stack(batch).to(device))
    j = 0
    for j1, j2 in batch_coords:
        results[j1, j2] = out[j]
        j += 1

    print(results.max())

    # ==== plotting ====
    import matplotlib.pyplot as plt
    from matplotlib import cm

    fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
    X, Y = np.meshgrid(coords1, coords2)
    ax.plot_surface(X, Y, results, cmap=cm.coolwarm,)
    # ax.scatter([1], [0], [1], c='black', linewidth=10)
    plt.savefig('loss.png')
    # plt.show()

    np.save('results', results)


@hydra.main(config_path="config/my_configs", config_name=CONFIG_NAME, version_base=None)
def main(cfg: DictConfig):
    # load data
    print("Dataset", cfg["dataset"]["name"])
    X_train, y_train, X_test, y_test = load_data(cfg["dataset"]["name"])
    dataset = MyDataset(X_test, y_test)
    object = dataset[0]

    device = torch.device(cfg["cuda"] if torch.cuda.is_available() else "cpu")

    model_path = os.path.join(
        cfg["model_folder"],
        f"model_{cfg['model_id']}_{cfg['dataset']['name']}.pt",
    )

    model = get_model(
        cfg["model"]["name"],
        cfg["model"]["params"],
        path=model_path,
        device=device,
    )

    visualize(
        dataset, 
        model, 
        cfg["grid_dim"], 
        cfg["batch_size"], 
        scale=cfg["scale"], 
        device=device,
    )

if __name__ == "__main__":
    main()
