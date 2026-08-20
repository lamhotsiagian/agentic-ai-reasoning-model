/**
 * Client for the reasoning API and the lab endpoints.
 *
 * Note the asymmetry that Chapter 1 argues for: `streamReasoning` consumes
 * *progress events* (a whitelisted projection), while the lab endpoints
 * return the full trajectory. Production UIs use the former; the labs exist
 * to show what the former deliberately hides.
 */
import { BACKEND_URL, fetchWithAuth } from './api';

export type StepKind =
  | 'decompose' | 'plan' | 'retrieve' | 'tool'
  | 'reason' | 'verify' | 'replan' | 'answer';

export interface ReasoningStep {
  step_id: string;
  kind: StepKind;
  thought: string;
  payload: Record<string, unknown>;
  observation: string | null;
  ok: boolean;
  error: string | null;
  latency_ms: number;
  prompt_tokens: number;
  completion_tokens: number;
}

export interface Trajectory {
  run_id: string;
  goal: string;
  difficulty: 'easy' | 'medium' | 'hard';
  tier: 'fast' | 'standard' | 'deep';
  steps: ReasoningStep[];
  final_answer: string | null;
  halted_reason: string | null;
  citations: string[];
}

export interface RunMetrics {
  run_id: string;
  tier: string;
  steps: number;
  tokens: number;
  tool_calls: number;
  retrieval_hops: number;
  replans: number;
  elapsed_ms: number;
  halted_reason: string | null;
  answered: boolean;
}

/** The ONLY reasoning shape that crosses the network in production. */
export interface ProgressEvent {
  step: string;
  label: string;
  status: 'ok' | 'retrying' | 'failed';
  duration_ms: number;
}

export interface AnswerEvent {
  step: 'answer' | 'error';
  content: string;
  citations?: string[];
  partial?: boolean;
  halted_reason?: string | null;
  metrics?: RunMetrics & { tier: string; gate: string };
}

export interface LabResult {
  [key: string]: unknown;
  trajectory?: Trajectory;
  metrics?: RunMetrics;
  progress?: ProgressEvent[];
}

export async function runLab(
  lab: number,
  prompt: string,
  config: Record<string, unknown> = {},
): Promise<LabResult> {
  const res = await fetchWithAuth('/reasoning/labs/run', {
    method: 'POST',
    body: JSON.stringify({ lab, prompt, config }),
  });
  if (!res.ok) {
    throw new Error(`lab ${lab} failed: ${res.status} ${await res.text()}`);
  }
  return res.json();
}

/** Consume the ndjson progress stream from the production route. */
export async function streamReasoning(
  threadId: string,
  prompt: string,
  onEvent: (e: ProgressEvent | AnswerEvent) => void,
  opts: { taskClass?: string; tier?: 'fast' | 'standard' | 'deep' } = {},
): Promise<void> {
  const token =
    typeof window !== 'undefined' ? localStorage.getItem('access_token') : null;

  const res = await fetch(`${BACKEND_URL}/reasoning/${threadId}`, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
      ...(token ? { Authorization: `Bearer ${token}` } : {}),
    },
    body: JSON.stringify({
      prompt,
      task_class: opts.taskClass ?? null,
      tier_override: opts.tier ?? null,
    }),
  });

  if (!res.body) throw new Error('no response body');
  const reader = res.body.getReader();
  const decoder = new TextDecoder();
  let buffer = '';

  for (;;) {
    const { done, value } = await reader.read();
    if (done) break;
    buffer += decoder.decode(value, { stream: true });
    const lines = buffer.split('\n');
    buffer = lines.pop() ?? '';
    for (const line of lines) {
      if (!line.trim()) continue;
      try {
        onEvent(JSON.parse(line));
      } catch {
        /* a partial line; the next chunk completes it */
      }
    }
  }
}

export const STEP_COLOR: Record<string, string> = {
  decompose: 'var(--accent)',
  plan: 'var(--accent)',
  retrieve: '#0e9488',
  tool: '#d97706',
  reason: '#4f46e5',
  verify: '#059669',
  replan: '#dc2626',
  answer: '#0f172a',
};

export function fmtMs(ms: number): string {
  return ms < 1000 ? `${ms}ms` : `${(ms / 1000).toFixed(1)}s`;
}
