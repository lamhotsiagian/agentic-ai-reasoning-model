'use client';

import { useState } from 'react';
import { ChevronDown, ChevronRight, AlertTriangle, Check } from 'lucide-react';
import { fmtMs, STEP_COLOR, type ReasoningStep, type RunMetrics, type Trajectory } from '@/lib/reasoning';

/**
 * Renders a raw trajectory. This component is a LAB affordance: in production
 * the browser receives whitelisted progress events, never this (Chapter 1).
 */
export function TrajectoryView({ trajectory, metrics }: {
  trajectory?: Trajectory;
  metrics?: RunMetrics;
}) {
  const [open, setOpen] = useState<Record<string, boolean>>({});
  if (!trajectory) return null;

  return (
    <div className="card p-4">
      <div className="flex items-center justify-between mb-3">
        <h3 className="font-semibold text-sm">Raw trajectory</h3>
        {metrics && <MetricsBar metrics={metrics} />}
      </div>

      {trajectory.halted_reason && (
        <div className="mb-3 flex items-start gap-2 rounded-lg border border-[var(--warning)] bg-[var(--warning-soft)] p-2 text-xs">
          <AlertTriangle className="w-4 h-4 text-[var(--warning)] shrink-0 mt-0.5" />
          <span>
            Halted: <strong>{trajectory.halted_reason}</strong>. The answer
            below is partial, not a conclusion.
          </span>
        </div>
      )}

      <ol className="space-y-1.5">
        {trajectory.steps.map((step, i) => (
          <StepRow
            key={step.step_id}
            index={i}
            step={step}
            open={!!open[step.step_id]}
            onToggle={() =>
              setOpen((s) => ({ ...s, [step.step_id]: !s[step.step_id] }))
            }
          />
        ))}
      </ol>
    </div>
  );
}

function StepRow({ index, step, open, onToggle }: {
  index: number; step: ReasoningStep; open: boolean; onToggle: () => void;
}) {
  const colour = STEP_COLOR[step.kind] ?? 'var(--muted)';
  return (
    <li className="rounded-lg border border-[var(--border)] overflow-hidden">
      <button
        onClick={onToggle}
        className="w-full flex items-center gap-2 px-2.5 py-1.5 text-left hover:bg-[var(--surface-muted)]"
      >
        {open ? <ChevronDown className="w-3.5 h-3.5 text-[var(--subtle)]" />
              : <ChevronRight className="w-3.5 h-3.5 text-[var(--subtle)]" />}
        <span className="text-[11px] text-[var(--subtle)] w-5">{index + 1}</span>
        <span
          className="badge"
          style={{ background: `${colour}14`, color: colour, borderColor: `${colour}44` }}
        >
          {step.kind}
        </span>
        <span className="text-xs text-[var(--muted)] truncate flex-1">
          {step.thought || step.observation || 'no detail'}
        </span>
        {step.ok
          ? <Check className="w-3.5 h-3.5 text-[var(--success)]" />
          : <AlertTriangle className="w-3.5 h-3.5 text-[var(--danger)]" />}
        <span className="text-[11px] text-[var(--subtle)] tabular-nums">
          {fmtMs(step.latency_ms)}
        </span>
        <span className="text-[11px] text-[var(--subtle)] tabular-nums">
          {step.prompt_tokens + step.completion_tokens}t
        </span>
      </button>

      {open && (
        <div className="border-t border-[var(--border)] bg-[var(--surface-sunken)] p-2.5 space-y-2">
          {step.thought && <Field label="thought" value={step.thought} />}
          {step.observation && <Field label="observation" value={step.observation} />}
          {step.error && <Field label="error" value={step.error} danger />}
          {Object.keys(step.payload).length > 0 && (
            <Field label="payload" value={JSON.stringify(step.payload, null, 2)} mono />
          )}
        </div>
      )}
    </li>
  );
}

function Field({ label, value, mono, danger }: {
  label: string; value: string; mono?: boolean; danger?: boolean;
}) {
  return (
    <div>
      <div className="field-label mb-0.5">{label}</div>
      <pre
        className={`text-[11px] whitespace-pre-wrap break-words ${
          mono ? 'font-mono' : ''
        } ${danger ? 'text-[var(--danger)]' : 'text-[var(--foreground)]'}`}
      >
        {value}
      </pre>
    </div>
  );
}

export function MetricsBar({ metrics }: { metrics: RunMetrics }) {
  const items: [string, string | number][] = [
    ['tier', metrics.tier],
    ['steps', metrics.steps],
    ['tokens', metrics.tokens],
    ['tools', metrics.tool_calls],
    ['hops', metrics.retrieval_hops],
    ['replans', metrics.replans],
    ['wall', fmtMs(metrics.elapsed_ms)],
  ];
  return (
    <div className="flex flex-wrap gap-1.5">
      {items.map(([k, v]) => (
        <span key={k} className="badge bg-[var(--surface-muted)] text-[var(--muted)] border-[var(--border)]">
          {k} <strong className="text-[var(--foreground)]">{v}</strong>
        </span>
      ))}
    </div>
  );
}
