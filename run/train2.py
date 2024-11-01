import sys
sys.path.append('/root/graduation_thetis/causal-bert-pytorch/')
from src.modelmodule.modelmodule import ImageCausalModel
import pandas as pd
import numpy as np
from src.datamodule.datamodule import CausalImageDataModule
from pytorch_lightning import Trainer
import torch
import hydra
from omegaconf import DictConfig
from pytorch_lightning.loggers import WandbLogger
from src.utils.common import set_seed, save_experiment_result
from sklearn.model_selection import train_test_split

@hydra.main(config_path = "conf", config_name = "train", version_base = '1.3')
def main(cfg:DictConfig):

    set_seed(cfg.seed)
    df = pd.read_csv("/root/graduation_thetis/causal-bert-pytorch/input/outputs_v4.csv")
    df["light_or_dark"] = df["light_or_dark"].apply(lambda x : 1 if x == "light" else 0)

    # データの分割
    train_df, temp_df = train_test_split(df, test_size=0.3, random_state=cfg.seed)
    val_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=cfg.seed)
    # インデックスのリセット
    train_df = train_df.reset_index(drop=True)
    val_df = val_df.reset_index(drop=True)
    test_df = test_df.reset_index(drop=True)
    
    # データモジュールの作成
    data_module = CausalImageDataModule(cfg, train_df, val_df, test_df)
    data_module.setup(stage = "fit")

    train_dataset_size = len(data_module.train_dataset)
    steps_per_epoch = train_dataset_size // cfg.batch_size
    total_training_steps = steps_per_epoch * cfg.epoch
    cfg.total_training_steps = total_training_steps

    model = ImageCausalModel(cfg)
    wandb_logger = WandbLogger(project='causal_image',
                               name = f'{cfg.pretrained_model}-{cfg.batch_size}-{cfg.epoch}', 
                               save_dir='logs')
    wandb_logger.experiment.config.update(dict(cfg))

    trainer = Trainer(
        logger = wandb_logger,
        max_epochs = cfg.epoch,
        accelerator='gpu' if torch.cuda.is_available() else 'cpu',
        enable_progress_bar = True,
        log_every_n_steps=cfg.total_training_steps // cfg.batch_size
    )

    trainer.fit(model, data_module)
    
    data_module.setup("test")

        # トレーニング後に保存
    #trainer.save_checkpoint("best_model.ckpt")

    # 別の場所で予測時に読み込み
    #model = ImageCausalModel.load_from_checkpoint("best_model.ckpt", cfg=cfg)

    trainer.predict(model, dataloaders=data_module.predict_dataloader())
    ATE_value = model.ate_value

    print(ATE_value)

    save_experiment_result(
        file_path=cfg.file_name,
        exp_name=f'exp{cfg.exp_id}',
        model_name=cfg.pretrained_model,
        batch_size=cfg.batch_size,
        epochs=cfg.epoch,
        ATE=float(ATE_value)
    )




if __name__ == '__main__':
    main()
