Q1: Why set up `projects/` folder?

Implementing new models and features into OpenMMLab's algorithm libraries could be troublesome due to the rigorous requirements on code quality, which could hinder the fast iteration of SOTA models and might discourage our members from sharing their latest outcomes here. And that's why we have this `projects/` folder now, where some experimental features, frameworks and models are placed, only needed to satisfy the minimum requirement on the code quality, and can be used as standalone libraries. Users are welcome to use them if they [use MMSegmentation from source](https://mmsegmentation.readthedocs.io/en/latest/get_started.html#best-practices).

Q2: Why should there be a checklist for a project?

This checkelist is crucial not only for this project's developers but the entire community, since there might be some other contributors joining this project and deciding their starting point from this list. It also helps maintainers accurately estimate time and effort on further code polishing, if needed.

Q3: What kind of PR will be merged?

Reaching the first milestone means that this project suffices the minimum requirement of being merged into 'projects/'. That is, the very first PR of a project must have all the terms in the first milestone checked. We do not have any extra requirements on the project's following PRs, so they can be a minor bug fix or update, and do not have to achieve one milestone at once. But keep in mind that this project is only eligible to become a part of the core package upon attaining the last milestone.

Q4: Compared to other models in the core packages, why do the model implementations in projects have different training/testing commands?

Projects are organized independently from the core package, and therefore their modules cannot be directly imported by train.py and test.py. Each model implementation in projects should either use `mim` for training/testing as suggested in the example project or provide a custom train.py/test.py.

Q5: How to debug a project with a debugger?

Debugger makes our lives easier, but using it becomes a bit tricky if we have to train/test a model via `mim`. The way to circumvent that is that we can take advantage of relative path to import these modules. Assuming that we are developing a project X and the core modules are placed under `projects/X/modules`, then simply adding `custom_imports = dict(imports='projects.X.modules')` to the config allows us to debug from usual entrypoints (e.g. `tools/train.py`) from the root directory of the algorithm library. Just don't forget to remove 'projects.X' before project publishment.
