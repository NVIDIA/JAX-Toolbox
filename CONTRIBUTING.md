# Contributing to JAX-Toolbox

Thank you for your interest in contributing to JAX-Toolbox! This document describes
how to propose changes and the requirements every contribution must meet before it
can be merged.

## How to contribute

1. **Open or find an issue.** For bug fixes and new features, please first search the
   [issue tracker](https://github.com/NVIDIA/JAX-Toolbox/issues) for an existing issue,
   or open a new one describing the problem or proposal. This helps us agree on the
   approach before you invest time in an implementation.
2. **Fork the repository and create a branch** from `main` for your change.
3. **Make your change.** Keep pull requests focused on a single topic, and include
   tests or documentation updates where appropriate.
4. **Run the pre-commit hooks.** The repository uses [pre-commit](https://pre-commit.com/)
   (see [`.pre-commit-config.yaml`](.pre-commit-config.yaml)) for whitespace, YAML and
   Python (`ruff`) checks:
   ```bash
   pip install pre-commit
   pre-commit install
   pre-commit run --all-files
   ```
5. **Sign off every commit** as described in [Signing your work](#signing-your-work).
6. **Open a pull request** against `main`. Describe what the change does and why, and
   link any related issues. A maintainer will review your pull request; CI for pull
   requests from forks is started by a maintainer after an initial review.

## License

JAX-Toolbox is dual-licensed (see [LICENSE.md](LICENSE.md)):

- **Source code** (for example Python, shell, Dockerfiles, CI workflows and other
  scripts or configuration) is licensed under the
  [Apache License 2.0](https://www.apache.org/licenses/LICENSE-2.0) (`Apache-2.0`).
- **Documentation, skills and other assets** (for example Markdown documentation,
  tutorials, agent skills and images) are licensed under the
  [Creative Commons Attribution 4.0 International](https://creativecommons.org/licenses/by/4.0/)
  license (`CC-BY-4.0`).

By contributing, you agree that your contribution is licensed under the terms that
apply to it: Apache-2.0 for source code and CC-BY-4.0 for documentation, skills and
assets.

New source files should start with an SPDX header (after the shebang line, if any):

```python
# SPDX-FileCopyrightText: Copyright (c) <year> NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
```

Documentation-style files are covered by the top-level [LICENSE.md](LICENSE.md). If a
documentation file or skill carries a license header, use `CC-BY-4.0`, for example
as an HTML comment in Markdown:

```markdown
<!--
SPDX-FileCopyrightText: Copyright (c) <year> NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: CC-BY-4.0
-->
```

For files with YAML front matter (such as agent skills' `SKILL.md`), put the header
after the front matter and the first heading, so that tools that parse the front
matter are not affected.

## Signing your work

JAX-Toolbox uses the [Developer Certificate of Origin (DCO)](DCO) for all
contributions. We require that every contributor "sign off" on each of their commits.
The sign-off certifies that the contribution is your original work, or that you have
the right to submit it under the license that applies to it in this project (see
[License](#license)) or a compatible license.

**Any contribution that contains commits that are not signed off will not be accepted.**

To sign off a commit, use the `--signoff` (or `-s`) option when committing:

```bash
git commit -s -m "Add cool feature."
```

This appends the following line to your commit message:

```
Signed-off-by: Your Name <your@email.com>
```

The name and email address must match your Git identity (`git config user.name` and
`git config user.email`). Pseudonymous or anonymous sign-offs cannot be accepted.

If you have already made commits without a sign-off, you can add it retroactively:

```bash
# Amend the most recent commit
git commit --amend --no-edit --signoff

# Sign off all commits on your branch since it diverged from main
git rebase --signoff main

git push --force-with-lease
```

### Developer Certificate of Origin

By signing off a commit, you certify the following (the full text is also available in
the [DCO](DCO) file):

```
Developer Certificate of Origin
Version 1.1

Copyright (C) 2004, 2006 The Linux Foundation and its contributors.

Everyone is permitted to copy and distribute verbatim copies of this
license document, but changing it is not allowed.


Developer's Certificate of Origin 1.1

By making a contribution to this project, I certify that:

(a) The contribution was created in whole or in part by me and I
    have the right to submit it under the open source license
    indicated in the file; or

(b) The contribution is based upon previous work that, to the best
    of my knowledge, is covered under an appropriate open source
    license and I have the right under that license to submit that
    work with modifications, whether created in whole or in part
    by me, under the same open source license (unless I am
    permitted to submit under a different license), as indicated
    in the file; or

(c) The contribution was provided directly to me by some other
    person who certified (a), (b) or (c) and I have not modified
    it.

(d) I understand and agree that this project and the contribution
    are public and that a record of the contribution (including all
    personal information I submit with it, including my sign-off) is
    maintained indefinitely and may be redistributed consistent with
    this project or the open source license(s) involved.
```

## Third-party code

If your contribution includes code, data or other files that you did not write
yourself (for example, code copied or adapted from another open source project),
please say so explicitly in the pull request description and include:

- where the material comes from (project name and URL),
- the license it is distributed under, and
- the original copyright and license notices, which must be preserved.

Material under a license that is not compatible with the license that applies to it
in this project (Apache-2.0 for source code, CC-BY-4.0 for documentation, skills and
assets) cannot be accepted.

## Information for NVIDIA contributors

In addition to the DCO sign-off above, NVIDIA employees must follow the NVIDIA IP
review process before any change is distributed publicly (that is, before it is pushed
to this repository or included in a release):
[https://nv/ip_review_process](https://nv/ip_review_process) (NVIDIA internal link).

In short, every change must be self-reviewed by the author and then peer-reviewed by a
reviewer from the team against the criteria in that process. The criteria apply to all
files in the project, including documentation, and cover, among other things:

- unfiled patentable subject matter,
- correct copyright headers,
- unannounced features or products,
- NVIDIA-proprietary algorithms or confidential internal information,
- disclosures that would give competitors a significant advantage, and
- third-party code that has not been cleared by the Open Source Review Board (OSRB).

If nobody on the team is qualified to review a particular item (for example, code that
comes from another team), engage that team directly to review it. Refer to the internal
page for the full criteria and for escalation to OSRB or IP Legal.

Pull requests from NVIDIA employees must also be signed off as described in
[Signing your work](#signing-your-work).
