import logging
import os

import cv2
import numpy as np
import torch
import tqdm
import yaml
from omegaconf import OmegaConf
from saicinpainting.evaluation.refinement import refine_predict
from saicinpainting.evaluation.utils import move_to_device
from saicinpainting.training.data.datasets import make_default_val_dataset
from saicinpainting.training.trainers import load_checkpoint
from torch.utils.data._utils.collate import default_collate

from inpaint import Inpainter
from src import directories

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
log = logging.getLogger(__name__)


class LamaInpainter(Inpainter):
    def __init__(
            self,
            abs_model_path: str,
            abs_input_dir: str,
            abs_output_dir: str,
            img_suffix: str,
    ):
        '''"""
        Initializes the instance variables of the class.

        Args:
            abs_model_path (str): The absolute path of the model.
            abs_input_dir (str): The absolute path of the input directory.
            abs_output_dir (str): The absolute path of the output directory.
            img_suffix (str): The suffix for the image files.
        """'''
        self.abs_model_path = abs_model_path
        self.abs_input_dir = abs_input_dir
        self.abs_output_dir = abs_output_dir
        self.img_suffix = img_suffix

    def inpaint(self):
        '''"""
        This method is used to perform inpainting on images. It first logs the absolute paths of the model, input directory, output directory, and image suffix. Then, it loads the default configuration file for prediction from 'lama-configs/prediction/default.yaml'. After that, it updates the model path, input directory, output directory, and image suffix in the configuration with the respective absolute paths. Finally, it calls the 'run_prediction' function with the updated configuration.

        Parameters:
        None

        Returns:
        The result of the 'run_prediction' function.
        """'''
        log.info(
            f"abs_model_path: {self.abs_model_path}, abs_input_dir: {self.abs_input_dir}, abs_output_dir: {self.abs_output_dir}, img_suffix: {self.img_suffix}"
        )
        from omegaconf import OmegaConf

        omega_conf = OmegaConf.load(os.path.join(directories.third_party_dir, "lama", "configs", "prediction", "default.yaml"))
        omega_conf.model.path = self.abs_model_path
        omega_conf.indir = self.abs_input_dir
        omega_conf.outdir = self.abs_output_dir
        omega_conf.dataset.img_suffix = self.img_suffix
        if torch.cuda.is_available():
            log.info("CUDA is available, using GPU")
            omega_conf.device = "cuda"
        else:
            log.info("CUDA is not available, using CPU")
            omega_conf.device = "cpu"
        return run_prediction(omega_conf)


def run_prediction(predict_config: OmegaConf) -> list[str]:
    '''"""
    This function runs the prediction process for a given configuration. It loads the model from the checkpoint, prepares the dataset, and runs the model on the dataset. The results are saved as images in the specified output directory.

    Args:
        predict_config (OmegaConf): The configuration object containing all the necessary parameters for the prediction process.

    Returns:
        list[str]: A list of file paths to the generated images.

    Raises:
        AssertionError: If the 'refine' option is set to True but 'unpad_to_size' is not in the batch.
    """'''
    generated_images = []
    if torch.cuda.is_available():
        log.info("CUDA is available, using GPU")
        predict_config.device = "cuda"
    device = torch.device(predict_config.device)
    train_config_path = os.path.join(predict_config.model.path, "config.yaml")
    with open(train_config_path, "r") as f:
        train_config = OmegaConf.create(yaml.safe_load(f))
    train_config.training_model.predict_only = True
    train_config.visualizer.kind = "noop"
    out_ext = predict_config.get("out_ext", ".png")
    checkpoint_path = os.path.join(
        predict_config.model.path, "models", predict_config.model.checkpoint
    )
    model = load_checkpoint(
        train_config, checkpoint_path, strict=False, map_location="cpu"
    )
    model.freeze()
    if not predict_config.get("refine", False):
        model.to(device)
    if not predict_config.indir.endswith("/"):
        predict_config.indir += "/"
    dataset = make_default_val_dataset(predict_config.indir, **predict_config.dataset)
    for img_i in tqdm.trange(len(dataset)):
        mask_fname = dataset.mask_filenames[img_i]
        cur_out_fname = os.path.join(
            predict_config.outdir,
            os.path.splitext(mask_fname[len(predict_config.indir):])[0] + out_ext,
        )
        os.makedirs(os.path.dirname(cur_out_fname), exist_ok=True)
        batch = default_collate([dataset[img_i]])
        if predict_config.get("refine", False):
            assert (
                    "unpad_to_size" in batch
            ), "Unpadded size is required for the refinement"
            cur_res = refine_predict(batch, model, **predict_config.refiner)
            cur_res = cur_res[0].permute(1, 2, 0).detach().cpu().numpy()
        else:
            with torch.no_grad():
                batch = move_to_device(batch, device)
                batch["mask"] = (batch["mask"] > 0) * 1
                batch = model(batch)
                cur_res = (
                    batch[predict_config.out_key][0]
                    .permute(1, 2, 0)
                    .detach()
                    .cpu()
                    .numpy()
                )
                unpad_to_size = batch.get("unpad_to_size", None)
                if unpad_to_size is not None:
                    (orig_height, orig_width) = unpad_to_size
                    cur_res = cur_res[:orig_height, :orig_width]
        cur_res = np.clip(cur_res * 255, 0, 255).astype("uint8")
        cur_res = cv2.cvtColor(cur_res, cv2.COLOR_RGB2BGR)
        cv2.imwrite(cur_out_fname, cur_res)
        generated_images.append(cur_out_fname)
        log.info(f"Saved {cur_out_fname}")
    return generated_images
