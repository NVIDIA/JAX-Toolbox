# JAX-Toolbox Documentation (Fern)

This directory contains the [Fern](https://buildwithfern.com) configuration for
the JAX-Toolbox documentation site. The layout mirrors the
[ai-dynamo/dynamo](https://github.com/ai-dynamo/dynamo) repository.

```
fern/
├── fern.config.json      # Fern org + pinned CLI version
├── docs.yml              # Site config: theme, logo, navbar, layout, mdx-components
└── README.md             # This file

docs/
├── _components/
│   └── EcosystemDiagram.tsx  # Interactive stack diagram (data injected by build script)
├── ecosystem/
│   ├── jax-on-nvidia-gpu-stack.mdx  # Page that mounts <EcosystemDiagram />
│   ├── stack.yml                    # Grid layout + project registry (source of truth)
│   └── projects/*.md                # Per-project description files
├── getting-started/      # Introduction, frameworks, build status, environment variables
├── reference/            # Container versions, staging containers, FAQ, clouds, resources
├── frameworks/           # MaxText, AXLearn
├── resiliency/           # Ray-based resilient JAX
├── inference/            # JAX-vLLM offloading bridge
└── *.md                  # Profiling, GPU performance, PGLE, FP8, nsys-jax, triage tool

scripts/
└── build-ecosystem.mjs   # Reads stack.yml + projects/*.md → splices DATA into EcosystemDiagram.tsx
```

## Local environment

The Fern CLI is managed as a **project-local** dev dependency — no global
install required. From the repository root:

```bash
nvm use        # optional: switch to the Node version pinned in .nvmrc (Node 22)
npm install    # installs fern-api into ./node_modules (one-time setup)
```

## npm scripts

All commands run from the **repository root**:

| Command | What it does |
|---|---|
| `npm run docs:dev` | Live preview at http://localhost:3000 (hot-reload on file changes) |
| `npm run docs:check` | Validate `docs.yml` / `index.yml` and check for broken links |
| `npm run docs:preview` | Build a shareable Fern preview deployment |
| `npm run docs:publish` | Build and deploy (requires a configured Fern token) |

Each command automatically runs `docs:ecosystem` first (see below), so the
interactive stack diagram is always up to date.

## Ecosystem diagram build step

`EcosystemDiagram.tsx` contains a `// ECOSYSTEM-DATA:START … // ECOSYSTEM-DATA:END`
block that is **generated** — never edit it by hand. Run the build script to
regenerate it:

```bash
npm run docs:ecosystem
```

The script reads `docs/ecosystem/stack.yml` (grid layout + project registry)
and `docs/ecosystem/projects/*.md` (per-project description text), then splices
a TypeScript `const DATA` block into `docs/_components/EcosystemDiagram.tsx`.

### Editing the diagram

**To add or update a project:**

1. Add or edit the project entry in `docs/ecosystem/stack.yml` under `projects:`.
2. Create or update `docs/ecosystem/projects/<id>.md` with a heading and
   description paragraph.
3. Place the project ID in the appropriate row `cells:` entry in `stack.yml`.
4. Run `npm run docs:ecosystem` (or just `npm run docs:dev`, which does it
   automatically).

**To change column widths or row order**, edit the `columns:` and `rows:`
sections in `stack.yml`. The grid is 12 units wide; column `width` values must
sum to 12.

## Notes

- `fern.config.json` pins the CLI version. Run `npx fern upgrade` to bump it;
  this rewrites the `version` field automatically.
- The `instances.url` in `docs.yml` is set to the Fern preview instance. Replace
  it with the production custom domain (`docs.nvidia.com/jax-toolbox/`) when
  publishing officially.
- MDX components are registered in `docs.yml` under
  `experimental.mdx-components` and must also be explicitly imported in each
  `.mdx` file that uses them:
  ```mdx
  import { EcosystemDiagram } from "../_components/EcosystemDiagram"
  ```
- The repository root `README.md` remains the GitHub landing page; the pages
  under `docs/getting-started/` are the docs-site equivalents.
