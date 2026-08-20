'use client';

import { useEffect, useRef, useState } from 'react';
import {
  Check, X, AlertTriangle, Loader2, ChevronDown, ChevronRight,
  Search, GitBranch, Wrench, ShieldCheck, Brain, Zap, Target, Play,
} from 'lucide-react';
import type { LabEvent } from '@/lib/useLabStream';
import { fmtMs } from '@/lib/reasoning';

const PHASE_META: Record<string, { icon: typeof Brain; color: string; label: string }> = {
  start:          { icon: Play,        color: '#6366f1', label: 'Start' },
  fast_path:      { icon: Zap,         color: '#f59e0b', label: 'Fast Path' },
  decompose:      { icon: GitBranch,   color: '#6366f1', label: 'Decompose' },
  plan:           { icon: Target,      color: '#6366f1', label: 'Plan' },
  retrieve:       { icon: Search,      color: '#0e9488', label: 'Retrieve' },
  tool:           { icon: Wrench,      color: '#d97706', label: 'Tool' },
  tool_call:      { icon: Wrench,      color: '#d97706', label: 'Tool Call' },
  reason:         { icon: Brain,       color: '#4f46e5', label: 'Reason' },
  verify:         { icon: ShieldCheck,  color: '#059669', label: 'Verify' },
  replan:         { icon: AlertTriangle,color: '#dc2626', label: 'Replan' },
  search:         { icon: Search,      color: '#0e9488', label: 'Search' },
  scope:          { icon: Wrench,      color: '#d97706', label: 'Scope' },
  fuse:           { icon: GitBranch,   color: '#0e9488', label: 'Fuse' },
  generate:       { icon: Brain,       color: '#4f46e5', label: 'Generate' },
  config:         { icon: Target,      color: '#6366f1', label: 'Config' },
  sweep:          { icon: Zap,         color: '#f59e0b', label: 'Sweep' },
  consistency:    { icon: ShieldCheck,  color: '#059669', label: 'Consistency' },
  tier_decision:  { icon: Target,      color: '#6366f1', label: 'Tier Decision' },
  supervisor:     { icon: Brain,       color: '#4f46e5', label: 'Supervisor' },
  specialist:     { icon: Wrench,      color: '#d97706', label: 'Specialist' },
  single_agent:   { icon: Brain,       color: '#4f46e5', label: 'Single Agent' },
  solve:          { icon: Brain,       color: '#4f46e5', label: 'Solve' },
  eval:           { icon: ShieldCheck,  color: '#059669', label: 'Evaluate' },
  node_run:       { icon: Wrench,      color: '#d97706', label: 'Node' },
  validate:       { icon: ShieldCheck,  color: '#059669', label: 'Validate' },
  chaos:          { icon: AlertTriangle,color: '#dc2626', label: 'Chaos' },
  deliberate_done:{ icon: Check,       color: '#059669', label: 'Complete' },
  answer:         { icon: Brain,       color: '#0f172a', label: 'Answer' },
  compose:        { icon: Brain,       color: '#4f46e5', label: 'Compose' },
};

// Also handle dynamic rung_* phases and channel_* phases
function getPhaseMeta(phase: string) {
  if (PHASE_META[phase]) return PHASE_META[phase];
  if (phase.startsWith('rung_')) return { icon: ShieldCheck, color: '#059669', label: phase.replace('rung_', 'Rung: ') };
  if (phase.startsWith('channel_')) return { icon: Search, color: '#0e9488', label: phase.replace('channel_', '') };
  return { icon: Brain, color: '#94a3b8', label: phase };
}

function StatusIcon({ status }: { status: string }) {
  switch (status) {
    case 'started':
      return <Loader2 className="w-3.5 h-3.5 animate-spin" style={{ color: 'var(--accent)' }} />;
    case 'completed':
      return <Check className="w-3.5 h-3.5" style={{ color: 'var(--success)' }} />;
    case 'failed':
      return <X className="w-3.5 h-3.5" style={{ color: 'var(--danger)' }} />;
    case 'retrying':
      return <AlertTriangle className="w-3.5 h-3.5" style={{ color: 'var(--warning)' }} />;
    default:
      return null;
  }
}

function EventCard({ event, isLast }: { event: LabEvent; isLast: boolean }) {
  const [expanded, setExpanded] = useState(false);
  const meta = getPhaseMeta(event.phase);
  const Icon = meta.icon;
  const hasData = Object.keys(event.data).length > 0;

  // Don't render the "done" event as a card; it's the final result
  if (event.phase === 'done' || event.phase === 'error') return null;

  return (
    <div className="flex gap-3 animate-timeline-in">
      {/* Timeline connector */}
      <div className="flex flex-col items-center shrink-0 w-8">
        <div
          className={`w-7 h-7 rounded-full flex items-center justify-center border-2 shrink-0 ${
            event.status === 'started' ? 'animate-timeline-pulse' : ''
          }`}
          style={{
            borderColor: meta.color,
            background: event.status === 'completed' ? `${meta.color}18` : 'var(--surface)',
          }}
        >
          <Icon className="w-3.5 h-3.5" style={{ color: meta.color }} />
        </div>
        {!isLast && (
          <div className="w-0.5 flex-1 min-h-[16px]" style={{ background: 'var(--border)' }} />
        )}
      </div>

      {/* Card */}
      <div className="flex-1 pb-3 min-w-0">
        <div
          className={`rounded-lg border overflow-hidden ${
            event.status === 'started'
              ? 'border-[var(--accent-border)] bg-[var(--accent-soft)]'
              : event.status === 'failed'
              ? 'border-[var(--danger-border)] bg-[var(--danger-soft)]'
              : 'border-[var(--border)] bg-[var(--surface)]'
          }`}
        >
          <button
            onClick={() => hasData && setExpanded(!expanded)}
            className={`w-full flex items-center gap-2 px-3 py-2 text-left ${
              hasData ? 'hover:bg-[var(--surface-muted)] cursor-pointer' : 'cursor-default'
            }`}
          >
            <StatusIcon status={event.status} />
            <span
              className="badge text-[10px]"
              style={{ background: `${meta.color}14`, color: meta.color, borderColor: `${meta.color}44` }}
            >
              {meta.label}
            </span>
            <span className="text-xs text-[var(--foreground)] truncate flex-1">
              {event.label}
            </span>
            <span className="text-[11px] text-[var(--subtle)] tabular-nums shrink-0">
              {fmtMs(event.elapsed_ms)}
            </span>
            {hasData && (
              expanded
                ? <ChevronDown className="w-3.5 h-3.5 text-[var(--subtle)] shrink-0" />
                : <ChevronRight className="w-3.5 h-3.5 text-[var(--subtle)] shrink-0" />
            )}
          </button>

          {expanded && hasData && (
            <div className="border-t border-[var(--border)] bg-[var(--surface-sunken)] p-2.5">
              <PhaseData phase={event.phase} data={event.data} />
            </div>
          )}
        </div>
      </div>
    </div>
  );
}

/** Render phase-specific data payloads in a readable format. */
function PhaseData({ phase, data }: { phase: string; data: Record<string, unknown> }) {
  // For certain phases, render structured output
  if (phase === 'fast_path' && data.answer) {
    return (
      <div className="space-y-1">
        <p className="text-xs whitespace-pre-wrap">{String(data.answer)}</p>
        <div className="flex gap-1.5">
          {data.tokens != null && (
            <span className="badge bg-[var(--surface-muted)] text-[var(--muted)] border-[var(--border)]">
              {String(data.tokens)} tokens
            </span>
          )}
          {data.wall_ms != null && (
            <span className="badge bg-[var(--surface-muted)] text-[var(--muted)] border-[var(--border)]">
              {fmtMs(data.wall_ms as number)}
            </span>
          )}
        </div>
      </div>
    );
  }

  if ((phase === 'decompose' || phase === 'decomposition') && data.dag) {
    const dag = data.dag as { sub_problems: { id: string; question: string; kind: string }[] };
    return (
      <ul className="space-y-1">
        {dag.sub_problems?.map((sp) => (
          <li key={sp.id} className="text-[11px] flex items-center gap-1.5">
            <span className="badge bg-[var(--accent-soft)] text-[var(--accent-text)] border-[var(--accent-border)]">
              {sp.id}
            </span>
            <span className="text-[var(--subtle)]">{sp.kind}</span>
            <span className="truncate">{sp.question}</span>
          </li>
        ))}
      </ul>
    );
  }

  if (phase.startsWith('channel_') && data.results) {
    const results = data.results as { doc: string; chunk: string; score: number; text: string }[];
    return (
      <ul className="space-y-1">
        {results.slice(0, 3).map((r, i) => (
          <li key={i} className="text-[11px]">
            <code className="text-[var(--accent-text)]">{r.doc}#{r.chunk}</code>
            <span className="text-[var(--subtle)] ml-1">{r.score.toFixed(3)}</span>
            <p className="text-[var(--muted)] line-clamp-1">{r.text}</p>
          </li>
        ))}
        {results.length > 3 && (
          <li className="text-[11px] text-[var(--subtle)]">+{results.length - 3} more</li>
        )}
      </ul>
    );
  }

  if (phase === 'validate' && data.plan) {
    const plan = data.plan as { steps: { sub_problem_id: string; tool: string | null }[] };
    return (
      <ol className="space-y-0.5">
        {plan.steps?.map((s, i) => (
          <li key={i} className="text-[11px] flex gap-1.5">
            <span className="text-[var(--subtle)]">{i + 1}.</span>
            <span className="badge bg-[var(--accent-soft)] text-[var(--accent-text)] border-[var(--accent-border)]">
              {s.sub_problem_id}
            </span>
            <code>{s.tool ?? 'n/a'}</code>
          </li>
        ))}
      </ol>
    );
  }

  if (phase.startsWith('rung_')) {
    return (
      <p className="text-[11px] text-[var(--muted)]">
        {data.detail ? String(data.detail) : 'Check completed'}
      </p>
    );
  }

  // Verdict data
  if (data.verdict) {
    const v = data.verdict as { passed: boolean; detail: string; score: number };
    return (
      <div className="text-[11px] space-y-0.5">
        <p>{v.passed ? '✓ Passed' : '✗ Failed'}: {v.detail}</p>
        {v.score != null && <p className="text-[var(--subtle)]">Score: {v.score}</p>}
      </div>
    );
  }

  // Answer/observation text
  if (data.answer) {
    return <p className="text-xs whitespace-pre-wrap line-clamp-4">{String(data.answer)}</p>;
  }
  if (data.observation) {
    return <p className="text-xs whitespace-pre-wrap line-clamp-3 text-[var(--muted)]">{String(data.observation)}</p>;
  }

  // Generic fallback: compact JSON
  const filtered = Object.fromEntries(
    Object.entries(data).filter(([, v]) => v != null && v !== '' && !(Array.isArray(v) && v.length === 0)),
  );
  if (Object.keys(filtered).length === 0) return null;
  return (
    <pre className="text-[11px] font-mono whitespace-pre-wrap break-words max-h-40 overflow-auto">
      {JSON.stringify(filtered, null, 2)}
    </pre>
  );
}

export function ReasoningTimeline({
  events,
  status,
  className,
}: {
  events: LabEvent[];
  status: 'idle' | 'running' | 'done' | 'error';
  className?: string;
}) {
  const bottomRef = useRef<HTMLDivElement>(null);

  // Auto-scroll to bottom as new events arrive
  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
  }, [events.length]);

  if (events.length === 0) return null;

  // Filter out the "done" event from the visual timeline
  const visible = events.filter((e) => e.phase !== 'done' && e.phase !== 'error');

  return (
    <div className={`card p-4 ${className ?? ''}`}>
      <div className="flex items-center justify-between mb-3">
        <h3 className="font-semibold text-sm flex items-center gap-2">
          {status === 'running' && <Loader2 className="w-4 h-4 animate-spin text-[var(--accent)]" />}
          {status === 'done' && <Check className="w-4 h-4 text-[var(--success)]" />}
          {status === 'error' && <X className="w-4 h-4 text-[var(--danger)]" />}
          Reasoning Process
        </h3>
        {events.length > 0 && (
          <span className="text-[11px] text-[var(--subtle)] tabular-nums">
            {visible.length} steps · {fmtMs(events[events.length - 1]?.elapsed_ms ?? 0)}
          </span>
        )}
      </div>

      <div className="space-y-0">
        {visible.map((event, i) => (
          <EventCard key={event.step_index} event={event} isLast={i === visible.length - 1 && status !== 'running'} />
        ))}
        {status === 'running' && visible.length > 0 && (
          <div className="flex gap-3 animate-timeline-in">
            <div className="flex flex-col items-center w-8">
              <div className="w-2 h-2 rounded-full bg-[var(--accent)] animate-timeline-pulse" />
            </div>
            <span className="text-[11px] text-[var(--subtle)] italic">Processing…</span>
          </div>
        )}
      </div>
      <div ref={bottomRef} />
    </div>
  );
}
