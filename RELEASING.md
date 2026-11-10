# Releasing

Releases are published to PyPI by `.github/workflows/publish.yml` when a `v*` tag
is pushed. It uses PyPI Trusted Publishing (OIDC), so no API token is stored in
the repository.

## One-time setup (project owner)

1. On <https://pypi.org>, add a publisher for the distribution
   `black_scholes_model`. Until the project exists, use a *pending publisher*
   (Account settings -> Publishing -> "Add a new pending publisher"); afterwards
   use the project's Publishing page.
2. Choose **GitHub** and fill in exactly:

   | Field            | Value                 |
   | ---------------- | --------------------- |
   | Owner            | `buzludeniz`          |
   | Repository name  | `black_scholes_model` |
   | Workflow name    | `publish.yml`         |
   | Environment name | `pypi`                |

3. Save. The `pypi` GitHub environment is created on the first run; optionally
   add required reviewers to it in the repository settings for a manual approval
   gate before upload.

A mismatch in any of these fields fails the upload with `invalid-publisher`.

## Cutting a release

1. Pick the version `X.Y.Z`. Update it in **both** `pyproject.toml` (`version`)
   and `src/black_scholes/__init__.py` (`__version__`);
   `tests/test_packaging.py` fails if they differ.
2. In `CHANGELOG.md`, rename `[Unreleased]` content into a new `[X.Y.Z] - YYYY-MM-DD`
   section, leave an empty `[Unreleased]` above it, and update the compare links
   at the bottom.
3. Open a PR, get CI green, merge to `main`.
4. Verify locally (optional, nothing is uploaded):

   ```bash
   python -m pip install build twine
   python -m build
   twine check --strict dist/*
   ```

5. Tag the merge commit and push the tag:

   ```bash
   git checkout main && git pull
   git tag -a vX.Y.Z -m "vX.Y.Z"
   git push origin vX.Y.Z
   ```

6. Watch the **Publish** workflow. The `build` job fails fast if the tag
   (`vX.Y.Z`, with the `v` stripped) differs from `version` in `pyproject.toml`,
   then builds, smoke-tests the wheel in a clean venv, and uploads the artifacts;
   the `publish` job uploads them to PyPI.
7. Create a GitHub release from the tag (optional) using the changelog section.

## Notes

- PyPI never allows re-uploading a filename. If a release is wrong, yank it and
  publish a new patch version.
- Do not run the workflow manually (`workflow_dispatch`) from a branch: the tag
  check is skipped for non-tag refs while the publish job still runs.
