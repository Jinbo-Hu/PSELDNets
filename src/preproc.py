import hydra
from omegaconf import DictConfig
from utils.config import get_dataset
from preproc.preprocess import Preprocess


@hydra.main(version_base="1.3", config_path="../configs", config_name="preproc.yaml")
def main(cfg: DictConfig):
    dataset = get_dataset(dataset_name=cfg.dataset, cfg=cfg)
    preprocessor = Preprocess(cfg, dataset)
    if cfg.mode == 'extract_data':
        # extract data and its index
        preprocessor.extract_index()
        if cfg.dataset == 'L3DAS22':
            preprocessor.extract_l3das22_label()

        if cfg.dataset_type == 'eval' and cfg.dataset == 'STARSS23':
            # 'STARSS23' dataset has no label of evaluation sets
            return
        # extract labels
        preprocessor.extract_accdoa_label()
        preprocessor.extract_track_label()
        preprocessor.extract_adpit_label()


if __name__ == "__main__":
    main()
