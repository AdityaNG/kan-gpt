Changelog
=========


(unreleased)
------------
- Version increment. [Aditya NG]
- Fix(kan_gpt/efficient_kan/__init__.py): missing init file. [Aditya NG]
- Feat(README.md): results added. [Aditya NG]
- Docs(mkdocs): documentation for mkdocs. [Aditya NG]
- Feat(sweep): slower learning rate. [Aditya NG]
- Fix(sweep): max_iters. [Aditya NG]
- Fix(sweep): cuda device. [Aditya NG]
- Ci(main): disabled wandb in ci. [Aditya NG]
- Test(tests/test_train.py): disable wandb during testing. [Aditya NG]
- Fix(sweep): added cuda clean. [Aditya NG]
- Feat(sweep): reduced batch size. [Aditya NG]
- Feat(download_dataset): functions to download tinyshakespeare and
  webtext. [Aditya NG]


0.3.0 (2024-05-07)
------------------
- Release: version 0.3.0 🚀 [Aditya NG]
- Docs(README.md): dataset mention. [Aditya NG]
- Feat(sweep): sweep script for getting a vast hyperparam sweep. [Aditya
  NG]
- Feat(tinyshakespeare): support for another dataset. [Aditya NG]
- Test(tests/test_prompt.py): eval code. [Aditya NG]
- Test(efficient_kan,original_kan): coverage. [Aditya NG]


0.2.0 (2024-05-04)
------------------
- Release: version 0.2.0 🚀 [Aditya NG]
- Version increment. [Aditya NG]
- Docs(KAN_GPT.ipynb): benchmark runs between mlp and kan gpts. [Aditya
  NG]
- Feat(train): date id of asset. [Aditya NG]
- Feat(train,prompt): saving model after training, prompting a model
  given the saved model path. [Aditya NG]
- Feat(kan_gpt/dataset.py): lengthwise dataset iteration. [Aditya NG]
- Fix(dataset): index error. [Aditya NG]


0.1.3 (2024-05-04)
------------------
- Release: version 0.1.3 🚀 [Aditya NG]
- Fix(dataset): padding and range fix. [Aditya NG]


0.1.2 (2024-05-04)
------------------
- Release: version 0.1.2 🚀 [Aditya NG]
- Feat(kan_gpt/efficient_kan/): integration with efficient kan reduced
  training memory footprint. [Aditya NG]
- Test(gpt-pico): reducing memory footprint. [Aditya NG]


0.1.1 (2024-05-04)
------------------
- Release: version 0.1.1 🚀 [Aditya NG]
- Test(kan_gpt,mlp_gpt,kan): test cases for forward-backward passes.
  [Aditya NG]
- Ci(codecov): token. [Aditya NG]
- Merge branch 'main' of https://github.com/AdityaNG/kan-gpt into main.
  [Aditya NG]
- Update LICENSE. [Aditya]
- Refactor(kan_gpt): linting. [Aditya NG]
- Merge pull request #5 from
  AdityaNG/dependabot/github_actions/actions/setup-python-5. [Aditya]

  Bump actions/setup-python from 4 to 5
- Bump actions/setup-python from 4 to 5. [dependabot[bot]]

  Bumps [actions/setup-python](https://github.com/actions/setup-python) from 4 to 5.
  - [Release notes](https://github.com/actions/setup-python/releases)
  - [Commits](https://github.com/actions/setup-python/compare/v4...v5)

  ---
  updated-dependencies:
  - dependency-name: actions/setup-python
    dependency-type: direct:production
    update-type: version-update:semver-major
  ...
- Merge pull request #4 from
  AdityaNG/dependabot/github_actions/codecov/codecov-action-4. [Aditya]

  Bump codecov/codecov-action from 3 to 4
- Bump codecov/codecov-action from 3 to 4. [dependabot[bot]]

  Bumps [codecov/codecov-action](https://github.com/codecov/codecov-action) from 3 to 4.
  - [Release notes](https://github.com/codecov/codecov-action/releases)
  - [Changelog](https://github.com/codecov/codecov-action/blob/main/CHANGELOG.md)
  - [Commits](https://github.com/codecov/codecov-action/compare/v3...v4)

  ---
  updated-dependencies:
  - dependency-name: codecov/codecov-action
    dependency-type: direct:production
    update-type: version-update:semver-major
  ...
- Merge pull request #3 from
  AdityaNG/dependabot/github_actions/actions/checkout-4. [Aditya]

  Bump actions/checkout from 3 to 4
- Bump actions/checkout from 3 to 4. [dependabot[bot]]

  Bumps [actions/checkout](https://github.com/actions/checkout) from 3 to 4.
  - [Release notes](https://github.com/actions/checkout/releases)
  - [Changelog](https://github.com/actions/checkout/blob/main/CHANGELOG.md)
  - [Commits](https://github.com/actions/checkout/compare/v3...v4)

  ---
  updated-dependencies:
  - dependency-name: actions/checkout
    dependency-type: direct:production
    update-type: version-update:semver-major
  ...
- Merge pull request #2 from
  AdityaNG/dependabot/github_actions/stefanzweifel/git-auto-commit-
  action-5. [Aditya]

  Bump stefanzweifel/git-auto-commit-action from 4 to 5
- Bump stefanzweifel/git-auto-commit-action from 4 to 5.
  [dependabot[bot]]

  Bumps [stefanzweifel/git-auto-commit-action](https://github.com/stefanzweifel/git-auto-commit-action) from 4 to 5.
  - [Release notes](https://github.com/stefanzweifel/git-auto-commit-action/releases)
  - [Changelog](https://github.com/stefanzweifel/git-auto-commit-action/blob/master/CHANGELOG.md)
  - [Commits](https://github.com/stefanzweifel/git-auto-commit-action/compare/v4...v5)

  ---
  updated-dependencies:
  - dependency-name: stefanzweifel/git-auto-commit-action
    dependency-type: direct:production
    update-type: version-update:semver-major
  ...
- Merge pull request #1 from
  AdityaNG/dependabot/github_actions/softprops/action-gh-release-2.
  [Aditya]

  Bump softprops/action-gh-release from 1 to 2
- Bump softprops/action-gh-release from 1 to 2. [dependabot[bot]]

  Bumps [softprops/action-gh-release](https://github.com/softprops/action-gh-release) from 1 to 2.
  - [Release notes](https://github.com/softprops/action-gh-release/releases)
  - [Changelog](https://github.com/softprops/action-gh-release/blob/master/CHANGELOG.md)
  - [Commits](https://github.com/softprops/action-gh-release/compare/v1...v2)

  ---
  updated-dependencies:
  - dependency-name: softprops/action-gh-release
    dependency-type: direct:production
    update-type: version-update:semver-major
  ...


0.1.0 (2024-05-03)
------------------
- Release: version 0.1.0 🚀 [Aditya NG]
- Funding. [Aditya NG]
- Feat(KAN): device type_as use. [Aditya NG]
- Feat(dataset.py): tqdm. [Aditya NG]
- Feat(kan_gpt/dataset.py): progress bar. [Aditya NG]
- Feat(train.py): wandb support. [Aditya NG]
- Feat(KAN_GPT.ipynb): notebook added for colab training. [Aditya NG]
- Linting. [Aditya NG]
- Training script improved. [Aditya NG]
- Initial commit. [Aditya NG]
- ✅ Ready to clone and code. [AdityaNG]
- Initial commit. [Aditya]


