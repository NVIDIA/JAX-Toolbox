# JAX-Toolbox Documentation (Fern)

This directory contains the [Fern](https://buildwithfern.com) configuration for
the JAX-Toolbox documentation site.

```
fern/
├── fern.config.json      # Fern org + pinned CLI version
├── docs.yml              # Site config: theme, logo, navbar, layout, mdx-components
└── README.md             # This file

docs/
├── _components/
│   ├── EcosystemDiagram.tsx  # Interactive stack diagram component
│   └── stack.ts              # Grid layout + project registry (edit this to update the diagram)
├── ecosystem/
│   └── jax-on-nvidia-gpu-stack.mdx  # Page that mounts <EcosystemDiagram />
├── getting-started/      # Introduction, frameworks, build status, environment variables
├── reference/            # Container versions, staging containers, FAQ, clouds, resources
├── frameworks/           # MaxText, AXLearn
├── resiliency/           # Ray-based resilient JAX
├── inference/            # JAX-vLLM offloading bridge
└── *.md                  # Profiling, GPU performance, PGLE, FP8, nsys-jax, triage tool
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

## Editing the ecosystem stack diagram

The diagram is driven by `docs/_components/stack.ts` — a plain TypeScript
object with three sections:

**`columns`** — the three vertical bands. Width values must sum to 12.

**`rows`** — horizontal layers from top (frameworks) to bottom (hardware).
Each row has `cells`, each cell has a `width` and a list of project `id`s.
Cell widths per row must also sum to 12.

**`projects`** — the registry of every node in the diagram:

```ts
{
  id: "myproject",           // unique; referenced in row cells
  name: "My Project",        // display name
  category: "nvidia",        // "nvidia" | "jax" | "other"
  href: "https://...",       // optional link shown in the detail panel
  nvidia_participates: true, // optional; adds green ring to jax/other nodes
  description: `One to three sentences shown when the user clicks the node.`,
},
```

Set `isStatic: true` (and omit `description`) for nodes that are not
clickable — used for the hardware row.

## Notes

- `fern.config.json` pins the CLI version. Run `npx fern upgrade` to bump it;
  this rewrites the `version` field automatically.
- The `instances.url` in `docs.yml` is set to the Fern preview instance. Replace
  it with the production custom domain (`docs.nvidia.com/jax-toolbox/`) when
  publishing officially.
- MDX components must be explicitly imported in each `.mdx` file that uses them:
  ```mdx
  import { EcosystemDiagram } from "../_components/EcosystemDiagram"
  ```
- The repository root `README.md` remains the GitHub landing page; the pages
  under `docs/getting-started/` are the docs-site equivalents.
