/**
 * Hook for streaming lab execution. Consumes the NDJSON progress stream
 * from POST /reasoning/labs/stream and manages incremental state.
 */
import { useCallback, useRef, useState } from 'react';
import { BACKEND_URL } from './api';
import type { LabResult } from './reasoning';

export interface LabEvent {
  phase: string;
  label: string;
  status: 'started' | 'completed' | 'failed' | 'retrying';
  data: Record<string, unknown>;
  elapsed_ms: number;
  step_index: number;
}

export interface LabStreamState {
  status: 'idle' | 'running' | 'done' | 'error';
  events: LabEvent[];
  currentPhase: string | null;
  result: LabResult | null;
  error: string | null;
}

const INITIAL_STATE: LabStreamState = {
  status: 'idle',
  events: [],
  currentPhase: null,
  result: null,
  error: null,
};

export function useLabStream() {
  const [state, setState] = useState<LabStreamState>(INITIAL_STATE);
  const abortRef = useRef<AbortController | null>(null);

  const run = useCallback(async (lab: number, prompt: string, config: Record<string, unknown> = {}) => {
    // Cancel any existing run
    abortRef.current?.abort();
    const controller = new AbortController();
    abortRef.current = controller;

    setState({ ...INITIAL_STATE, status: 'running' });

    try {
      const token = typeof window !== 'undefined' ? localStorage.getItem('access_token') : null;
      const res = await fetch(`${BACKEND_URL}/reasoning/labs/stream`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          ...(token ? { Authorization: `Bearer ${token}` } : {}),
        },
        body: JSON.stringify({ lab, prompt, config }),
        signal: controller.signal,
      });

      if (!res.ok) {
        const text = await res.text();
        throw new Error(`Lab ${lab} failed: ${res.status} ${text}`);
      }

      if (!res.body) throw new Error('No response body');
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
            const event: LabEvent = JSON.parse(line);
            setState((prev) => {
              const events = [...prev.events, event];
              if (event.phase === 'done') {
                return {
                  ...prev,
                  status: 'done',
                  events,
                  currentPhase: null,
                  result: (event.data as unknown) as LabResult,
                };
              }
              if (event.phase === 'error') {
                return {
                  ...prev,
                  status: 'error',
                  events,
                  currentPhase: null,
                  error: event.label,
                };
              }
              return {
                ...prev,
                events,
                currentPhase: event.status === 'started' ? event.phase : prev.currentPhase,
              };
            });
          } catch {
            /* partial JSON line; the next chunk will complete it */
          }
        }
      }

      // If we finished the stream without a 'done' event, mark as done
      setState((prev) => prev.status === 'running' ? { ...prev, status: 'done' } : prev);
    } catch (err) {
      if ((err as Error).name === 'AbortError') return;
      setState((prev) => ({
        ...prev,
        status: 'error',
        error: err instanceof Error ? err.message : String(err),
      }));
    }
  }, []);

  const cancel = useCallback(() => {
    abortRef.current?.abort();
    setState((prev) => ({ ...prev, status: prev.status === 'running' ? 'done' : prev.status }));
  }, []);

  const reset = useCallback(() => {
    abortRef.current?.abort();
    setState(INITIAL_STATE);
  }, []);

  return { state, run, cancel, reset };
}
