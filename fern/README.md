# JAX-Toolbox Documentation (Fern)

This directory contains the [Fern](https://buildwithfern.com) configuration for
the JAX-Toolbox documentation site. The layout mirrors the
[ai-dynamo/dynamo](https://github.com/ai-dynamo/dynamo) repository:

```
fern/
├── fern.config.json      # Fern org + pinned CLI version
├── docs.yml              # Site config: theme, logo, navbar, footer, layout + navigation
├── main.css              # Custom CSS overrides
├── components/
│   └── CustomFooter.tsx  # Optional custom React footer
└── README.md             # This file

docs/
├── assets/               # Logos, favicon, (optional) fonts
├── getting-started/      # Intro, frameworks, build status, environment variables
├── reference/            # Container versions, staging containers, FAQ, clouds, resources
├── frameworks/           # MaxText, AXLearn (existing content)
├── resiliency/           # Ray-based resilient JAX (existing content)
└── *.md                  # Profiling, GPU performance, PGLE, FP8, nsys-jax, triage tool
```

## Local environment

The Fern CLI is managed as a **project-local** dev dependency (the npm
equivalent of a Python venv) — no global install required. From the repository
root:

```bash
nvm use            # optional: use the Node version pinned in .nvmrc
npm install        # installs fern-api into ./node_modules (one time)
```

Everything then runs through npm scripts (defined in the root `package.json`),
which resolve the local `node_modules/.bin/fern`:

```bash
npm run docs:dev       # live preview at http://localhost:3000
npm run docs:check     # validate docs.yml / index.yml and check links
npm run docs:preview   # build a shareable preview deployment
npm run docs:publish   # build & deploy (requires a configured Fern instance/token)
```

You can also invoke the local CLI directly with `npx fern <command>`.

## Notes

- `fern.config.json` pins the CLI version. Run `fern upgrade` to bump it to the
  latest release; this rewrites the `version` field for you.
- The `instances.url` in `docs.yml` is a placeholder — replace it with the real
  JAX-Toolbox Fern instance URL and add the production custom domain.
- Logo/favicon SVGs under `docs/assets/` are the official NVIDIA brand assets
  (`NVIDIA_light.svg`, `NVIDIA_dark.svg`, `NVIDIA_symbol.svg`).
- The repository root `README.md` remains the GitHub landing page; the
  getting-started pages here are the docs-site equivalents.
