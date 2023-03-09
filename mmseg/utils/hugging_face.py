# Copyright (c) OpenMMLab. All rights reserved.
import importlib
import os
import os.path as osp
from functools import partial
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Optional, Tuple

from mmengine import Config
from mmengine.fileio import load
from mmengine.utils.dl_utils import load_url

try:
    from huggingface_hub import (create_repo, get_hf_file_metadata,
                                 hf_hub_download, hf_hub_url,
                                 repo_type_and_id_from_hf_id, upload_folder)
    from huggingface_hub.utils import EntryNotFoundError
    hf_hub_download = partial(
        hf_hub_download, library_name=None, library_version=None)
    _has_hf_hub = True
except ImportError:
    hf_hub_download = None
    _has_hf_hub = False
HF_WEIGHTS_NAME = 'pytorch_model.bin'  # default pytorch pkl


def has_hf_hub(necessary=False):
    if not _has_hf_hub and necessary:
        # if no HF Hub module installed, and it is necessary to continue,
        # raise error
        raise RuntimeError(
            'Hugging Face hub model specified but package not installed. '
            'Run `pip install huggingface_hub`.')
    return _has_hf_hub


def create_from_hf_hub(repo_id):

    from mmseg.apis import init_model

    # Download config from HF Hub
    config_file = hf_hub_download(repo_id=repo_id, filename='config.json')
    # Download ckpt from HF Hub
    checkpoint_file = hf_hub_download(
        repo_id=repo_id, filename=HF_WEIGHTS_NAME)
    return init_model(config=config_file, checkpoint=checkpoint_file)


def generate_readme(results_info: dict, model_name: str) -> str:
    """Generate README (model card for Hugging face Hub)

    Args:
        results_info (dict): The results information of model.
        model_name (str): The model name for the model card.

    Returns:
        str: The text readme for the model.
    """
    readme_text = '---\n'
    readme_text += 'language:\n- en\n'
    readme_text += 'license: apache-2.0\n'
    readme_text += 'library_name: mmsegmentation\n'
    readme_text += 'tags:\n- semantic segmentation\n- openmmlab\n'
    if results_info.get('Dataset'):
        readme_text += f'datasets:\n- {results_info["Dataset"]}'
    if results_info.get('Metrics'):
        readme_text += f'metrics:\n- {results_info["Metrics"]["mIoU"]}'

    readme_text += '---\n'

    readme_text += f'# Model card for {model_name}\n'
    # TODO: Add more description
    return readme_text


def push_to_hf_hub(model: str,
                   repo_id: str,
                   commit_message: Optional[str] = 'Add model',
                   token: Optional[str] = None,
                   revision: Optional[str] = None,
                   private: bool = False,
                   create_pr: bool = False) -> str:
    """Push model from MMSegmentation to Hugging face Hub.

    Args:
        model (str): The model which will be uploaded. It can be the model name
            or alias in metafile.
        repo_id (str): The repository to which the file will be uploaded, for
            example: `"username/custom_transformers"`.
        commit_message (str, optional): The summary / title / first line of the
            generated commit. Defaults to: 'Add model'.
        token (str, optional): Authentication token, obtained by `HfApi.login`
            method. Will default to the stored token. Defaults to None.
        revision (str, optional): The git revision to commit from. Defaults to
            None, i.e. the head of the `"main"` branch.
        private (bool, optional): Whether the model repo should be private.
            Defaults to False.
        create_pr (bool, optional): Whether or not to create a Pull Request
            with that commit. Defaults to `False`. If `revision` is not set,
            PR is opened against the `"main"` branch. If `revision` is set and
            is a branch, PR is opened against this branch. If `revision` is set
            and is not a branch name (example: a commit oid), an
            `RevisionNotFoundError` is returned by the server.

    Returns:
        str: A URL to visualize the uploaded folder on the hub
    """
    # Create repo if it doesn't exist yet
    repo_url = create_repo(
        repo_id, token=token, private=private, exist_ok=True)
    # Infer complete repo_id from repo_url
    # Can be different from the input `repo_id` if repo_owner was implicit
    _, repo_owner, repo_name = repo_type_and_id_from_hf_id(repo_url)
    repo_id = f'{repo_owner}/{repo_name}'

    # Check if README file already exist in repo
    try:
        get_hf_file_metadata(
            hf_hub_url(
                repo_id=repo_id, filename='README.md', revision=revision))
        has_readme = True
    except EntryNotFoundError:
        has_readme = False

    # Dump model and push to Hub
    with TemporaryDirectory() as tmpdir:
        # Save model weights and config.
        results_info = save_for_hf(model, tmpdir)

        # Add readme if it does not exist
        if not has_readme:
            model_name = repo_id.split('/')[-1]
            readme_path = Path(tmpdir) / 'README.md'
            readme_text = generate_readme(results_info, model_name)
            readme_path.write_text(readme_text)

        # Upload model and return
        return upload_folder(
            repo_id=repo_id,
            folder_path=tmpdir,
            revision=revision,
            create_pr=create_pr,
            commit_message=commit_message,
        )


def save_for_hf(model, save_directory) -> dict:
    """Save the files for Hugging face Hub.

    Args:
        model (str): The model which will be uploaded. It can be the model name
            or alias in metafile.
        save_directory (str): The directory to save the checkpont file and
            config file which will be uploaded to hub.

    Returns:
        ditc: The results information of model.
    """
    config_path = Path(save_directory) / 'config.json'
    model_path = Path(save_directory) / HF_WEIGHTS_NAME
    # input is model name
    config, ckpt, results_info = _load_model_from_metafile(model)
    config.dump(config_path)
    ckpt_org_name = osp.basename(ckpt)
    load_url(ckpt, Path(save_directory))
    os.rename(osp.join(Path(save_directory), ckpt_org_name), model_path)
    return results_info


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
    for model_meta in _get_models_from_metafile(repo_or_mim_dir):
        model_name = model_meta['Name'].lower()
        model_aliases = model_meta.get('Alias', [])
        if isinstance(model_aliases, str):
            model_aliases = [model_aliases.lower()]
        else:
            model_aliases = [alias.lower() for alias in model_aliases]
        if (model_name == model or model in model_aliases):
            cfg = Config.fromfile(
                osp.join(repo_or_mim_dir, model_meta['Config']))
            weights = model_meta['Weights']
            weights = weights[0] if isinstance(weights, list) else weights
            results_info = model_meta['Results']
            results_info = results_info[0] if isinstance(
                results_info, list) else results_info
            return cfg, weights, results_info
    raise ValueError(f'Cannot find model: {model} in mmsegmentation')


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
