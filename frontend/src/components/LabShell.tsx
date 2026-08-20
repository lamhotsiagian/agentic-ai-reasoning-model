'use client';

import { ReactNode, useState } from 'react';
import Link from 'next/link';
import { ArrowLeft, Play, Loader2, Square } from 'lucide-react';
import { fmtMs } from '@/lib/reasoning';

export interface LabMeta {
  id: number;
  chapter: string;
  title: string;
  question: string;      // the question the lab answers
  module: string;        // the backend module it exercises
  samples: string[];
}

export function LabShell({
  meta, controls, children, onRun, running, error, prompt, setPrompt,
  onCancel, elapsed,
}: {
  meta: LabMeta;
  controls?: ReactNode;
  children: ReactNode;
  onRun: () => void;
  running: boolean;
  error?: string | null;
  prompt: string;
  setPrompt: (v: string) => void;
  onCancel?: () => void;
  elapsed?: number;
}) {
  return (
    <div className="flex-1 overflow-y-auto p-6 md:p-8">
      <div className="max-w-6xl mx-auto space-y-5">
        <div>
          <Link href="/labs" className="btn btn-ghost text-xs px-2 py-1 -ml-2 mb-2">
            <ArrowLeft className="w-3.5 h-3.5" /> All labs
          </Link>
          <div className="flex items-baseline gap-3 flex-wrap">
            <span className="badge bg-[var(--accent-soft)] text-[var(--accent-text)] border-[var(--accent-border)]">
              Lab {meta.id}
            </span>
            <h1 className="text-2xl font-bold">{meta.title}</h1>
            <span className="text-xs text-[var(--subtle)]">{meta.chapter}</span>
          </div>
          <p className="text-[var(--muted)] text-sm mt-1.5 max-w-3xl">{meta.question}</p>
          <code className="text-[11px] text-[var(--subtle)]">{meta.module}</code>
        </div>

        <div className="card p-4 space-y-3">
          <div>
            <label className="field-label">Prompt</label>
            <textarea
              className="input mt-1 min-h-[76px] font-mono text-[13px]"
              value={prompt}
              onChange={(e) => setPrompt(e.target.value)}
              placeholder="Ask something…"
            />
          </div>

          <div className="flex flex-wrap gap-1.5">
            {meta.samples.map((s) => (
              <button
                key={s}
                onClick={() => setPrompt(s)}
                className="badge bg-[var(--surface-muted)] text-[var(--muted)] border-[var(--border)] hover:border-[var(--accent-border)] hover:text-[var(--accent-text)] max-w-full truncate"
                title={s}
              >
                {s.length > 72 ? `${s.slice(0, 72)}…` : s}
              </button>
            ))}
          </div>

          {controls && (
            <div className="border-t border-[var(--border)] pt-3 flex flex-wrap items-end gap-4">
              {controls}
            </div>
          )}

          <div className="flex items-center gap-3">
            <button className="btn btn-primary px-4 py-2 text-sm" onClick={onRun} disabled={running || !prompt.trim()}>
              {running ? <Loader2 className="w-4 h-4 animate-spin" /> : <Play className="w-4 h-4" />}
              {running ? 'Running…' : 'Run'}
            </button>
            {running && onCancel && (
              <button className="btn btn-secondary px-3 py-2 text-sm" onClick={onCancel}>
                <Square className="w-3.5 h-3.5" /> Cancel
              </button>
            )}
            {running && elapsed != null && (
              <span className="text-xs text-[var(--subtle)] tabular-nums">{fmtMs(elapsed)}</span>
            )}
            {error && <span className="text-xs text-[var(--danger)]">{error}</span>}
          </div>
        </div>

        {children}
      </div>
    </div>
  );
}

export function Toggle({ label, checked, onChange, hint }: {
  label: string; checked: boolean; onChange: (v: boolean) => void; hint?: string;
}) {
  return (
    <label className="flex items-start gap-2 cursor-pointer select-none">
      <input
        type="checkbox"
        checked={checked}
        onChange={(e) => onChange(e.target.checked)}
        className="mt-0.5 accent-[var(--accent)]"
      />
      <span>
        <span className="text-xs font-medium">{label}</span>
        {hint && <span className="block text-[11px] text-[var(--subtle)]">{hint}</span>}
      </span>
    </label>
  );
}

export function Slider({ label, value, onChange, min, max, step = 1, hint }: {
  label: string; value: number; onChange: (v: number) => void;
  min: number; max: number; step?: number; hint?: string;
}) {
  return (
    <div className="min-w-[170px]">
      <div className="flex justify-between">
        <span className="field-label">{label}</span>
        <span className="text-xs font-mono text-[var(--accent-text)]">{value}</span>
      </div>
      <input
        type="range" min={min} max={max} step={step} value={value}
        onChange={(e) => onChange(Number(e.target.value))}
        className="w-full accent-[var(--accent)]"
      />
      {hint && <span className="block text-[11px] text-[var(--subtle)]">{hint}</span>}
    </div>
  );
}

export function Select<T extends string>({ label, value, onChange, options, hint }: {
  label: string; value: T; onChange: (v: T) => void;
  options: readonly T[]; hint?: string;
}) {
  return (
    <div className="min-w-[150px]">
      <label className="field-label">{label}</label>
      <select
        className="input mt-1 py-1.5"
        value={value}
        onChange={(e) => onChange(e.target.value as T)}
      >
        {options.map((o) => <option key={o} value={o}>{o}</option>)}
      </select>
      {hint && <span className="block text-[11px] text-[var(--subtle)]">{hint}</span>}
    </div>
  );
}

export function Panel({ title, subtitle, children, tone }: {
  title: string; subtitle?: string; children: ReactNode;
  tone?: 'ok' | 'warn' | 'danger';
}) {
  const border =
    tone === 'ok' ? 'var(--success-border)'
    : tone === 'warn' ? 'var(--warning)'
    : tone === 'danger' ? 'var(--danger-border)'
    : 'var(--border)';
  return (
    <section className="card p-4" style={{ borderColor: border }}>
      <h3 className="font-semibold text-sm">{title}</h3>
      {subtitle && <p className="text-xs text-[var(--muted)] mt-0.5 mb-2">{subtitle}</p>}
      <div className="mt-2">{children}</div>
    </section>
  );
}
