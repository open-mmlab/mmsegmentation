# Copyright (c) OpenMMLab. All rights reserved.
import importlib
import json
import os.path as osp
from collections import OrderedDict
from functools import partial
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Optional, Tuple, Union

import torch
from mmengine import Config, ConfigDict
from mmengine.fileio import load
from mmengine.utils.dl_utils import load_url
from torch.nn import Module

ConfigType = Union[Config, ConfigDict]

try:
    from huggingface_hub import (create_repo, get_hf_file_metadata,
                                 hf_hub_download, hf_hub_url,
                                 repo_type_and_id_from_hf_id, upload_folder)
    from huggingface_hub.utils import EntryNotFoundError
    hf_hub_download = partial(
        hf_hub_download, library_name="openmmlab", library_version='2.0')
    _has_hf_hub = True
except ImportError:
    hf_hub_download = None
    _has_hf_hub = False
HF_WEIGHTS_NAME = "pytorch_model.bin"  # default pytorch pkl


def has_hf_hub(necessary=False):
    if not _has_hf_hub and necessary:
        # if no HF Hub module installed, and it is necessary to continue, raise error
        raise RuntimeError(
            'Hugging Face hub model specified but package not installed. Run `pip install huggingface_hub`.'
        )
    return _has_hf_hub


def generate_readme(model_card: dict, model_name: str):
    readme_text = "---\n"
    readme_text += "tags:\n- semantic segmentation\n- openmmlab\n"
    readme_text += "library_tag: openmmlab/mmsegmentation\n"
    readme_text += f"license: {model_card.get('license', 'apache-2.0')}\n"
    if 'details' in model_card and 'Dataset' in model_card['details']:
        readme_text += 'datasets:\n'
        readme_text += f"- {model_card['details']['Dataset'].lower()}\n"
        if 'Pretrain Dataset' in model_card['details']:
            readme_text += f"- {model_card['details']['Pretrain Dataset'].lower()}\n"
    readme_text += "---\n"
    readme_text += f"# Model card for {model_name}\n"
    if 'description' in model_card:
        readme_text += f"\n{model_card['description']}\n"
    if 'details' in model_card:
        readme_text += "\n## Model Details\n"
        for k, v in model_card['details'].items():
            if isinstance(v, (list, tuple)):
                readme_text += f"- **{k}:**\n"
                for vi in v:
                    readme_text += f"  - {vi}\n"
            elif isinstance(v, dict):
                readme_text += f"- **{k}:**\n"
                for ki, vi in v.items():
                    readme_text += f"  - {ki}: {vi}\n"
            else:
                readme_text += f"- **{k}:** {v}\n"
    if 'usage' in model_card:
        readme_text += f"\n## Model Usage\n"
        readme_text += model_card['usage']
        readme_text += '\n'

    if 'comparison' in model_card:
        readme_text += f"\n## Model Comparison\n"
        readme_text += model_card['comparison']
        readme_text += '\n'

    if 'citation' in model_card:
        readme_text += f"\n## Citation\n"
        if not isinstance(model_card['citation'], (list, tuple)):
            citations = [model_card['citation']]
        else:
            citations = model_card['citation']
        for c in citations:
            readme_text += f"```bibtex\n{c}\n```\n"
    return readme_text


def push_to_hf_hub(model: Union[str, ConfigType, Module],
                   repo_id: str,
                   commit_message: str = 'Add model',
                   token: Optional[str] = None,
                   revision: Optional[str] = None,
                   private: bool = False,
                   create_pr: bool = False,
                   model_config: Optional[dict] = None,
                   model_card: Optional[dict] = None):
    """_summary_

    Args:
        model (_type_): _description_
        repo_id (str): _description_
        commit_message (str, optional): _description_. Defaults to 'Add model'.
        token (Optional[str], optional): _description_. Defaults to None.
        revision (Optional[str], optional): _description_. Defaults to None.
        private (bool, optional): _description_. Defaults to False.
        create_pr (bool, optional): _description_. Defaults to False.
        model_config (Optional[dict], optional): _description_. Defaults to None.
        model_card (Optional[dict], optional): _description_. Defaults to None.

    Returns:
        _type_: _description_
    """
    # Create repo if it doesn't exist yet
    repo_url = create_repo(
        repo_id, token=token, private=private, exist_ok=True)
    # Infer complete repo_id from repo_url
    # Can be different from the input `repo_id` if repo_owner was implicit
    _, repo_owner, repo_name = repo_type_and_id_from_hf_id(repo_url)
    repo_id = f"{repo_owner}/{repo_name}"

    # Check if README file already exist in repo
    try:
        get_hf_file_metadata(
            hf_hub_url(
                repo_id=repo_id, filename="README.md", revision=revision))
        has_readme = True
    except EntryNotFoundError:
        has_readme = False

    # Dump model and push to Hub
    with TemporaryDirectory() as tmpdir:
        # Save model weights and config.
        save_for_hf(model, tmpdir)

        # Add readme if it does not exist
        if not has_readme:
            model_card = model_card or {}
            model_name = repo_id.split('/')[-1]
            readme_path = Path(tmpdir) / "README.md"
            readme_text = generate_readme(model_card, model_name)
            readme_path.write_text(readme_text)

        # Upload model and return
        return upload_folder(
            repo_id=repo_id,
            folder_path=tmpdir,
            revision=revision,
            create_pr=create_pr,
            commit_message=commit_message,
        )


def save_for_hf(model, save_directory):
    model_path = Path(save_directory) / HF_WEIGHTS_NAME
    config_path = Path(save_directory) / 'config.json'
    if isinstance(model, str):
        # input is model name
        config, ckpt = _load_model_from_metafile(model)
        load_url(ckpt, model_path)

        config.dump(config_path)

    elif isinstance(model, torch.nn.Module):
        # input is a torch module
        ckpt = model.state_dict()
        torch.save(ckpt, model_path)
    elif isinstance(model, OrderedDict):
        torch.save(ckpt, model_path)


def save_config_for_hf(config, save_directory):
    pass


def _load_model_from_metafile(model: str) -> Tuple[Config, str]:
    """Load config and weights from metafile.

        Args:
            model (str): model name defined in metafile.

        Returns:
            Tuple[Config, str]: Loaded Config and weights path defined in
            metafile.
        """
    model = model.lower()

    repo_or_mim_dir = _get_repo_or_mim_dir()
    for model_cfg in _get_models_from_metafile(repo_or_mim_dir):
        model_name = model_cfg['Name'].lower()
        model_aliases = model_cfg.get('Alias', [])
        if isinstance(model_aliases, str):
            model_aliases = [model_aliases.lower()]
        else:
            model_aliases = [alias.lower() for alias in model_aliases]
        if (model_name == model or model in model_aliases):
            cfg = Config.fromfile(
                osp.join(repo_or_mim_dir, model_cfg['Config']))
            weights = model_cfg['Weights']
            weights = weights[0] if isinstance(weights, list) else weights
            return cfg, weights
    raise ValueError(f'Cannot find model: {model} in {self.scope}')


def _get_models_from_metafile(dir: str):
    """Load model config defined in metafile from package path.

        Args:
            dir (str): Path to the directory of Config. It requires the
                directory ``Config``, file ``model-index.yml`` exists in the
                ``dir``.

        Yields:
            dict: Model config defined in metafile.
        """
    meta_indexes = load(osp.join(dir, 'model-index.yml'))
    for meta_path in meta_indexes['Import']:
        # meta_path example: mmcls/.mim/configs/conformer/metafile.yml
        meta_path = osp.join(dir, meta_path)
        metainfo = load(meta_path)
        yield from metainfo['Models']


def _get_repo_or_mim_dir():

    module = importlib.import_module('mmseg')
    # Since none of OpenMMLab series packages are namespace packages
    # (https://docs.python.org/3/glossary.html#term-namespace-package),
    # The first element of module.__path__ means package installation path.
    package_path = module.__path__[0]

    if osp.exists(osp.join(osp.dirname(package_path), 'configs')):
        repo_dir = osp.dirname(package_path)
        return repo_dir
    else:
        mim_dir = osp.join(package_path, '.mim')
        if not osp.exists(osp.join(mim_dir, 'Configs')):
            raise FileNotFoundError(
                f'Cannot find Configs directory in {package_path}!, '
                f'please check the completeness of the mmseg.')
        return mim_dir