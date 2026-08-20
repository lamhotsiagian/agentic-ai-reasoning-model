'use client';

import Link from 'next/link';
import { BookOpen, ChevronRight } from 'lucide-react';

/**
 * Lab index. One lab per chapter of *Agentic AI Reasoning Model System
 * Design*, each isolating a single mechanism so the behaviour can be
 * observed rather than taken on faith.
 */
export const LABS = [
  {
    id: 1, chapter: 'Chapter 1: Fundamentals',
    title: 'Fast path vs. deliberate path',
    desc: 'Run the same prompt through one forward pass and through the reasoning loop. Watch the budget halt a run and return a partial answer.',
    module: 'app/reasoning/state.py',
  },
  {
    id: 2, chapter: 'Chapter 2: Decomposition',
    title: 'The decomposition DAG',
    desc: 'See the sub-problem graph, its parallel layers, and what happens to the answer when global constraints are dropped.',
    module: 'app/reasoning/decomposer.py',
  },
  {
    id: 3, chapter: 'Chapter 3: Planning',
    title: 'Plan validation and replanning',
    desc: 'Reversible-first ordering, capability-gap rejection before execution, and the replan escalation ladder under injected chaos.',
    module: 'app/reasoning/planner.py',
  },
  {
    id: 4, chapter: 'Chapter 4: Search',
    title: 'The live search tree',
    desc: 'Best-of-N, beam, and Tree of Thoughts on the same problem. Pruned branches are greyed rather than hidden, because seeing what was discarded is the point.',
    module: 'app/reasoning/search.py',
  },
  {
    id: 5, chapter: 'Chapter 5: Tools',
    title: 'The tool workbench',
    desc: 'Registry scoping, typed errors, circuit breakers, and SQL AST validation against a keyword blocklist.',
    module: 'app/reasoning/tools/',
  },
  {
    id: 6, chapter: 'Chapter 6: Retrieval',
    title: 'Multi-hop hybrid retrieval',
    desc: 'Per-channel rankings side by side, RRF fusion, reranking, and the sufficiency gap statement that generates the next query.',
    module: 'app/reasoning/retrieval/',
  },
  {
    id: 7, chapter: 'Chapter 7: Verification',
    title: 'The verification cascade',
    desc: 'Catch an unfaithful trace by re-executing its own arithmetic, and watch self-reflection break answers it should have left alone.',
    module: 'app/reasoning/verify.py',
  },
  {
    id: 8, chapter: 'Chapter 8: Test-time compute',
    title: 'The scaling curve',
    desc: 'Sweep the budget and plot accuracy against tokens. Compare static and adaptive routing, and see adaptive-N cut sampling cost.',
    module: 'app/reasoning/compute.py',
  },
  {
    id: 9, chapter: 'Chapter 9: Multi-agent',
    title: 'Supervisor vs. single agent',
    desc: 'The blackboard diff after each specialist, and the experiment that matters: what happens when the critic is shown the reasoner’s trace.',
    module: 'app/reasoning/agents/',
  },
  {
    id: 10, chapter: 'Chapter 10: Production',
    title: 'The production console',
    desc: 'Progress events against the raw trajectory, the decomposed evaluation report, and guardrails catching an injection at two different layers.',
    module: 'app/reasoning/routes.py',
  },
] as const;

export default function LabsIndexPage() {
  return (
    <div className="flex-1 overflow-y-auto p-6 md:p-8">
      <div className="max-w-6xl mx-auto">
        <div className="mb-8">
          <h1 className="text-2xl font-bold flex items-center gap-3">
            <div className="p-2 bg-[var(--accent-soft)] border border-[var(--accent-border)] rounded-xl">
              <BookOpen className="text-[var(--accent)] w-5 h-5" />
            </div>
            Reasoning Labs
          </h1>
          <p className="text-[var(--muted)] text-sm mt-1 max-w-3xl">
            One lab per chapter. Each isolates a single mechanism and exposes the
            internals the production API deliberately hides, so you can watch the
            behaviour instead of trusting the prose.
          </p>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-5">
          {LABS.map((lab) => (
            <Link
              key={lab.id}
              href={`/labs/${lab.id}`}
              className="group card card-hover p-5 flex flex-col gap-2.5"
            >
              <div className="flex justify-between items-center">
                <span className="badge bg-[var(--accent-soft)] text-[var(--accent-text)] border-[var(--accent-border)]">
                  Lab {lab.id}
                </span>
                <ChevronRight className="w-4 h-4 text-[var(--subtle)] group-hover:text-[var(--accent)] group-hover:translate-x-0.5 transition" />
              </div>
              <div>
                <div className="text-[11px] text-[var(--subtle)]">{lab.chapter}</div>
                <h3 className="font-semibold group-hover:text-[var(--accent)] transition">
                  {lab.title}
                </h3>
              </div>
              <p className="text-[var(--muted)] text-sm leading-relaxed flex-1">{lab.desc}</p>
              <code className="text-[11px] text-[var(--subtle)]">{lab.module}</code>
            </Link>
          ))}
        </div>
      </div>
    </div>
  );
}
