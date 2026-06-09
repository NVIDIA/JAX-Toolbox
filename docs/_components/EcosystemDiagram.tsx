declare const React: any;

import { STACK } from "./stack";

// ---------------------------------------------------------------------------
// Type definitions
// ---------------------------------------------------------------------------

interface ColumnDef {
  id: string;
  label: string;
  width: number;
}

interface CellDef {
  width: number;
  label?: string;
  projects: string[];
}

interface RowDef {
  id: string;
  label: string;
  cells: CellDef[];
}

interface ProjectDef {
  id: string;
  name: string;
  category: string;
  href?: string;
  nvidia_participates?: boolean;
  isStatic?: boolean;
  description?: string;
}

// ---------------------------------------------------------------------------
const DATA = STACK as unknown as { columns: ColumnDef[]; rows: RowDef[]; projects: ProjectDef[] };

// ---------------------------------------------------------------------------
// Category styling — rgba tints so the diagram reads on both light and dark
// page backgrounds (same intent as the original color-mix() CSS approach)
// ---------------------------------------------------------------------------

const NV_GREEN = "#76B900";
const JAX_BLUE = "#5e9bf0";
const OTHER_GRAY = "#6b7280";

// background tint + inset ring color per category
const CAT_STYLE: Record<string, { bg: string; ring: string }> = {
  nvidia: { bg: "rgba(118,185,0,0.12)",   ring: NV_GREEN   },
  jax:    { bg: "rgba(94,155,240,0.12)",  ring: JAX_BLUE   },
  other:  { bg: "rgba(107,114,128,0.14)", ring: OTHER_GRAY },
};
const CAT_DEFAULT = { bg: "rgba(107,114,128,0.10)", ring: OTHER_GRAY };

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

function totalUnits(columns: ColumnDef[]): number {
  return columns.reduce((s, c) => s + c.width, 0);
}

function catStyle(project: ProjectDef): { bg: string; ring: string } {
  const base = CAT_STYLE[project.category] ?? CAT_DEFAULT;
  // nvidia_participates overrides the ring to green regardless of category
  if (project.nvidia_participates && project.category !== "nvidia") {
    return { bg: base.bg, ring: NV_GREEN };
  }
  return base;
}

// ---------------------------------------------------------------------------
// Legend
// ---------------------------------------------------------------------------

function Legend() {
  const items = [
    { bg: "rgba(118,185,0,0.12)",   ring: NV_GREEN,   label: "NVIDIA-developed / optimized" },
    { bg: "rgba(94,155,240,0.12)",  ring: JAX_BLUE,   label: "JAX-native / OSS ecosystem" },
    { bg: "rgba(107,114,128,0.14)", ring: OTHER_GRAY,  label: "Other / runtime" },
    { bg: "rgba(94,155,240,0.12)",  ring: NV_GREEN,   label: "NVIDIA participates" },
  ];
  return (
    <div style={{ display: "flex", flexWrap: "wrap", gap: "12px", margin: "0 0 10px", fontSize: "0.68rem" }}>
      {items.map((item) => (
        <span key={item.label} style={{ display: "inline-flex", alignItems: "center", gap: "6px" }}>
          <span
            style={{
              display: "inline-block",
              width: "14px",
              height: "11px",
              borderRadius: "1px",
              background: item.bg,
              boxShadow: `inset 0 0 0 1.5px ${item.ring}`,
            }}
          />
          {item.label}
        </span>
      ))}
    </div>
  );
}

// ---------------------------------------------------------------------------
// Column headers
// ---------------------------------------------------------------------------

function ColumnHeaders({ columns }: { columns: ColumnDef[] }) {
  return (
    <>
      {columns.map((col) => (
        <div
          key={col.id}
          style={{
            gridColumn: `span ${col.width}`,
            textAlign: "center",
            fontWeight: 700,
            fontSize: "0.64rem",
            padding: "3px 6px",
            borderRadius: "2px",
            background: "rgba(118,185,0,0.10)",
            border: "1px solid rgba(118,185,0,0.45)",
          }}
        >
          {col.label}
        </div>
      ))}
    </>
  );
}

// ---------------------------------------------------------------------------
// Project node (chip)
// ---------------------------------------------------------------------------

function ProjectNode({
  project,
  selected,
  onClick,
}: {
  project: ProjectDef;
  selected: boolean;
  onClick: () => void;
}) {
  const { bg, ring } = catStyle(project);
  const isClickable = !project.isStatic;

  return (
    <button
      onClick={isClickable ? onClick : undefined}
      style={{
        display: "inline-block",
        background: bg,
        boxShadow: selected
          ? `0 0 0 2px ${NV_GREEN}, inset 0 0 0 2px ${ring}`
          : `inset 0 0 0 2px ${ring}`,
        outline: "none",
        border: "none",
        borderRadius: "2px",
        fontWeight: selected ? 700 : 600,
        fontSize: "0.82rem",
        lineHeight: 1.2,
        padding: "4px 9px",
        color: "inherit",
        cursor: isClickable ? "pointer" : "default",
        textDecoration: "none",
        whiteSpace: "nowrap",
      }}
      aria-pressed={selected || undefined}
    >
      {project.name}
    </button>
  );
}

// ---------------------------------------------------------------------------
// Grid cell
// ---------------------------------------------------------------------------

function Cell({
  cell,
  projectMap,
  selectedId,
  onSelect,
}: {
  cell: CellDef;
  projectMap: Map<string, ProjectDef>;
  selectedId: string | null;
  onSelect: (id: string) => void;
}) {
  const hasLabel = !!cell.label;
  return (
    <div
      style={{
        gridColumn: `span ${cell.width}`,
        position: "relative",
        display: "flex",
        flexWrap: "wrap",
        gap: "7px 5px",
        justifyContent: "center",
        alignContent: "center",
        alignItems: "center",
        padding: hasLabel ? "1.5rem 5px 8px" : "7px 5px 8px",
        border: "1px dashed rgba(128,128,128,0.32)",
        borderRadius: "2px",
        minHeight: "1.8rem",
      }}
    >
      {hasLabel && (
        <span
          style={{
            position: "absolute",
            top: "3px",
            left: "5px",
            fontSize: "0.5rem",
            fontWeight: 700,
            letterSpacing: "0.04em",
            textTransform: "uppercase",
            opacity: 0.55,
            pointerEvents: "none",
          }}
        >
          {cell.label}
        </span>
      )}
      {cell.projects.map((pid) => {
        const proj = projectMap.get(pid);
        if (!proj) return null;
        return (
          <ProjectNode
            key={pid}
            project={proj}
            selected={selectedId === pid}
            onClick={() => onSelect(pid)}
          />
        );
      })}
    </div>
  );
}

// ---------------------------------------------------------------------------
// Overview panel
// ---------------------------------------------------------------------------

function OverviewPanel({
  project,
  onClose,
}: {
  project: ProjectDef;
  onClose: () => void;
}) {
  const { ring } = catStyle(project);
  return (
    <div
      style={{
        marginTop: "0.5rem",
        border: `1px solid rgba(118,185,0,0.4)`,
        borderTop: `3px solid ${NV_GREEN}`,
        borderRadius: "5px",
        background: "rgba(118,185,0,0.06)",
        padding: "0.9rem 1.1rem",
        fontSize: "0.85rem",
        position: "relative",
      }}
    >
      <button
        onClick={onClose}
        style={{
          position: "absolute",
          top: "8px",
          right: "10px",
          background: "transparent",
          border: "none",
          opacity: 0.5,
          fontSize: "16px",
          cursor: "pointer",
          lineHeight: 1,
          padding: "0 4px",
          color: "inherit",
        }}
        aria-label="Close"
      >
        ×
      </button>
      <strong style={{ display: "block", marginBottom: "0.4rem" }}>
        <span
          style={{
            display: "inline-block",
            background: "rgba(0,0,0,0.08)",
            boxShadow: `inset 0 0 0 2px ${ring}`,
            borderRadius: "2px",
            padding: "2px 7px",
            marginRight: "8px",
            fontSize: "0.82rem",
          }}
        >
          {project.name}
        </span>
        {project.nvidia_participates && (
          <span style={{ fontSize: "0.75rem", color: NV_GREEN }}>
            NVIDIA participates
          </span>
        )}
      </strong>
      {project.description && (
        <p style={{ margin: "0 0 0.5rem", lineHeight: 1.55 }}>{project.description}</p>
      )}
      {project.href && (
        <a
          href={project.href}
          target="_blank"
          rel="noopener noreferrer"
          style={{ color: NV_GREEN, fontWeight: 600, textDecoration: "none", fontSize: "0.82rem" }}
        >
          Learn more →
        </a>
      )}
    </div>
  );
}

// ---------------------------------------------------------------------------
// Main component
// ---------------------------------------------------------------------------

export function EcosystemDiagram() {
  const [selectedId, setSelectedId] = React.useState(null);

  const projectMap = React.useMemo(
    () => new Map(DATA.projects.map((p: ProjectDef) => [p.id, p])),
    []
  );

  function handleSelect(id: string) {
    setSelectedId((prev: string | null) => (prev === id ? null : id));
  }

  const selectedProject = selectedId ? projectMap.get(selectedId) : null;
  const units = totalUnits(DATA.columns);

  return (
    <div style={{ margin: "1.25rem 0" }}>
      <Legend />
      <div
        role="group"
        aria-label="JAX on NVIDIA GPU stack"
        style={{
          display: "grid",
          gridTemplateColumns: `repeat(${units}, 1fr)`,
          gap: "4px",
          fontSize: "0.78rem",
          overflowX: "auto",
          minWidth: "560px",
        }}
      >
        <ColumnHeaders columns={DATA.columns} />
        {DATA.rows.map((row: RowDef) =>
          row.cells.map((cell, ci) => (
            <Cell
              key={`${row.id}-${ci}`}
              cell={cell}
              projectMap={projectMap}
              selectedId={selectedId}
              onSelect={handleSelect}
            />
          ))
        )}
      </div>
      {selectedProject ? (
        <OverviewPanel
          project={selectedProject}
          onClose={() => setSelectedId(null)}
        />
      ) : (
        <div
          style={{
            marginTop: "0.5rem",
            border: "1px solid rgba(118,185,0,0.4)",
            borderTop: `3px solid ${NV_GREEN}`,
            borderRadius: "5px",
            background: "rgba(118,185,0,0.06)",
            padding: "0.9rem 1.1rem",
            fontSize: "0.85rem",
            fontStyle: "italic",
            opacity: 0.55,
          }}
        >
          Select a project to see what it is and where it fits.
        </div>
      )}
    </div>
  );
}
