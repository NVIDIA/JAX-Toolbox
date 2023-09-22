# Development
Rosetta builds upon other base libraries like [t5x](https://github.com/google-research/t5x)
and [paxml](https://github.com/google/paxml) and patches it with features
that are necessary to train the models defined in this repo.

This project follows the [distribution model](https://opensource.com/article/18/7/forks-vs-distributions). This means
we cherry-pick the necessary changes onto the tip of upstream's `main` branch. These changes are typically available as PRs
against upstream.

So before committing a change to this repo, ask yourself: "Does this change need to keep up with
upstream?" For example, if you want to add a new flag to [t5x/train.py](https://github.com/google-research/t5x/blob/main/t5x/train.py),
it makes more sense to create a PR for this and cherry-pick it when forming the t5x distribution.
This way we don't maintain a copy of `t5x/train.py` with that change, since it can get stale
very quickly.

Note: The reason to create a PR, is that even if the change is not accepted, or the original
branch along with the fork is deleted, the PR and its metadata will still exist for us to
pull.

For all other changes like adding a new model or a new data pipeline library that is orthogonal to
the base library: we can add that to this repo.

## Changes that need to keep up with upstream
**Note**: Here we use [t5x](https://github.com/google-research/t5x) as an example, but the same applies to [paxml](https://github.com/google/paxml).

The workflow to create a feature that we'll cherry-pick into the t5x distribution is:

2. Create a [fork](https://github.com/google-research/t5x/fork) with your changes
3. Create a PR with that fork against upstream ([example](https://github.com/google-research/t5x/pull/1204))
4. Add that PR to [patchlist-t5x.txt](../patchlist-t5x.txt) as a line (for above example add `pull/1204/head`)
5. Create a PR in this repo to update patchlist-t5x.txt

It is possible that by adding your change to the `patchlist.txt`, you create a merge conflict
when cherry-picking. It is for that reason that when you create a new PR for the t5x
distribution, you need to make sure:

1. The change is atomic
2. It is not dependent on other PRs

Understandably, the above two conditions are very difficult to adhere to always. If your change
conflicts with a previous PR, message the PR author or the maintainers of rosetta on how to
proceed. Most likely, it may mean breaking up/restructing your change, or adding your change
on top of an existing PR's branch.

## Creating the T5X distribution

### Docker
If you build rosetta's Dockerfile, you will automatically build the distribution.
You can build the project's image with or without the dev dependencies (used for testing/linting):
```bash
ROSETTA_BASE=pax  # or t5x

docker buildx build --target rosetta --tag rosetta:latest -f Dockerfile.${ROSETTA_BASE} .

# If you want a devel image with test dependencies
docker buildx build --target rosetta-devel --tag rosetta-devel:latest -f Dockerfile.${ROSETTA_BASE} .

# If you want to specify a specific base image
docker buildx build --target rosetta --tag rosetta:latest -f Dockerfile.${ROSETTA_BASE} --build-arg BASE_IMAGE=ghcr.io/nvidia/${ROSETTA_BASE}:nightly-2023-05-01 .
```

### Locally
**Note**: Here we use [t5x](https://github.com/google-research/t5x) as an example, but the same applies to [paxml](https://github.com/google/paxml).

If you need to rebuild the t5x distribution locally to work thru a merge conflict,
you can do the following:
```bash
# The "INSTALLED_T5X_DIR" is assumed to be the upstream repo
INSTALLED_T5X_DIR=/tmp/t5x
git clone https://github.com/google-research/t5x.git $INSTALLED_T5X_DIR

# --patchlist: is the aforementioned patchlist.txt 
# --mirror-url: Git url of a mirror of the upstream repo to obtain pull requests or branches
# --dir: is the directory where t5x is installed
# --ref: is optional and can be set to a commit hash or a git-ref in case you want to build the
#        distribution from a commit other than the tip of main.
./create-distribution.sh --patchlist patchlist-t5x.txt --mirror-url https://github.com/google-research/t5x.git --dir $INSTALLED_T5X_DIR --ref 79909538d7d98a46966cc683ec7fa606b0f7cf78
```

Afterwards, you can look at the logs and the state of `$INSTALLED_T5X_DIR` to see how you'd like
to resolve the MR.

## Linting and Testing
This repo uses pytest for testing and ruff for linting. To install those the necessary libraries, you can
use the extras:
```bash
pip install -e '.[lint,test]'
```

To lint this project, just run the following at the root:
```bash
ruff .

# And to ask ruff to fix what it can
ruff --fix .
```

To run the test cases in this project run (assumes 2 GPUs are available):
```bash
pytest -n3 .

# To run just integration tests (turned off by default because they take longer)
pytest -n3 -m integration .
```
