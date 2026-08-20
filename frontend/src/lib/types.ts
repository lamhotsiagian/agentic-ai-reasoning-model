export interface User {
  id: string;
  username: string;
  email: string;
  first_name?: string;
  last_name?: string;
}

export interface Thread {
  id: string;
  title: string;
  user_id: string;
  created_at: string;
}

export interface Memory {
  id: string;
  user_id: string;
  thread_id?: string;
  memory_type: 'episodic' | 'semantic' | 'procedural' | 'entity';
  content: string;
  importance_score: number;
  access_count: number;
  last_accessed_at: string;
  created_at: string;
  decay_rate: number;
  is_active: boolean;
  is_shared: boolean;
  metadata_json: string;
}

export interface Message {
  role: 'human' | 'ai' | 'system' | 'tool';
  content: string;
}

export interface MemoryConflict {
  id: string;
  user_id: string;
  memory_id_old: string;
  memory_id_new: string;
  old_content: string;
  new_content: string;
  conflict_type: string;
  resolution?: string;
  is_resolved: boolean;
  created_at: string;
}

export interface MemoryStats {
  total_count: number;
  counts_by_type: Record<string, number>;
  avg_importance: number;
  active_count: number;
  inactive_count: number;
}

/* ---------------------------------------------------------------------------
   Reasoning layer. The public shapes only. The raw trajectory type lives in
   lib/reasoning.ts because it is a LAB affordance, not a production contract.
   --------------------------------------------------------------------------- */

export type ReasoningTier = 'fast' | 'standard' | 'deep';

export interface ReasoningRequest {
  prompt: string;
  task_class?: string | null;
  strategy?: 'auto' | 'cot' | 'least_to_most' | 'react' | 'hybrid' | 'search';
  tier_override?: ReasoningTier | null;
  max_hops?: number;
}

export interface ToolDescriptor {
  name: string;
  description: string;
  tags: string[];
  reversibility: 0 | 1 | 2 | 3;
  args: Record<string, unknown>;
}

export interface Budget {
  max_steps: number;
  max_tokens: number;
  max_wall_ms: number;
  max_tool_calls: number;
}
