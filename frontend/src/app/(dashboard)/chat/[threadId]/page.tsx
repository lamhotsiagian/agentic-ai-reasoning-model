'use client';

import { useParams } from 'next/navigation';
import { useEffect, useState, useRef } from 'react';
import { fetchWithAuth, BACKEND_URL } from '@/lib/api';
import { Message } from '@/lib/types';
import { Send, Loader, User, Cpu, Upload, Check, BrainCircuit, Terminal } from 'lucide-react';

export default function ChatWindowPage() {
  const params = useParams();
  const threadId = params.threadId as string;

  const [messages, setMessages] = useState<Message[]>([]);
  const [input, setInput] = useState('');
  const [loading, setLoading] = useState(false);
  const [streamingContent, setStreamingContent] = useState('');

  const [uploading, setUploading] = useState(false);
  const [uploadNotice, setUploadNotice] = useState('');
  const fileInputRef = useRef<HTMLInputElement>(null);

  const [toolLogs, setToolLogs] = useState<string[]>([]);
  const messagesEndRef = useRef<HTMLDivElement>(null);
  const [uploadedDocs, setUploadedDocs] = useState<any[]>([]);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages, streamingContent, toolLogs]);

  const loadHistory = async () => {
    try {
      const res = await fetchWithAuth(`/chat/${threadId}`);
      if (res.ok) {
        const history = await res.json();
        setMessages(
          history.map((m: { role: string; content: string }) => ({
            role: m.role === 'human' ? 'human' : 'ai',
            content: m.content,
          }))
        );
      }
    } catch (e) {
      console.error(e);
    }
  };

  const loadDocuments = async () => {
    try {
      const res = await fetchWithAuth(`/documents/${threadId}`);
      if (res.ok) {
        const docs = await res.json();
        setUploadedDocs(docs);
      }
    } catch (e) {
      console.error(e);
    }
  };

  useEffect(() => {
    if (threadId) {
      loadHistory();
      loadDocuments();
      setToolLogs([]);
      setStreamingContent('');
    }
  }, [threadId]);

  const handleSend = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!input.trim() || loading) return;

    const userMessage: Message = { role: 'human', content: input };
    setMessages((prev) => [...prev, userMessage]);
    setInput('');
    setLoading(true);
    setToolLogs([]);
    setStreamingContent('');

    // Local accumulator, which avoids the stale-closure bug where the final
    // assistant message was saved from an empty state value.
    let accumulated = '';

    try {
      const token = localStorage.getItem('access_token');
      const res = await fetch(`${BACKEND_URL}/chat/${threadId}`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          Authorization: `Bearer ${token}`,
        },
        body: JSON.stringify({ prompt: userMessage.content, model_name: 'llama3.1' }),
      });

      if (!res.ok) throw new Error('Streaming connection failed');

      const reader = res.body?.getReader();
      const decoder = new TextDecoder();
      if (!reader) return;

      let buffer = '';
      while (true) {
        const { value, done } = await reader.read();
        if (done) break;

        buffer += decoder.decode(value, { stream: true });
        const lines = buffer.split('\n');
        buffer = lines.pop() || '';

        for (const line of lines) {
          if (!line.trim()) continue;
          try {
            const data = JSON.parse(line);
            if (data.type === 'llm_chunk') {
              accumulated += data.content;
              setStreamingContent(accumulated);
            } else if (data.type === 'tool_call') {
              setToolLogs((prev) => [...prev, `Called ${data.name}${data.args?.query ? `: "${data.args.query}"` : ''}`]);
            } else if (data.type === 'tool_result') {
              setToolLogs((prev) => [...prev, `Result: ${data.content?.slice(0, 80) ?? ''}…`]);
            }
          } catch {
            // ignore malformed line fragments
          }
        }
      }

      setMessages((prev) => [...prev, { role: 'ai', content: accumulated || '(No response received.)' }]);
      setStreamingContent('');
    } catch {
      setMessages((prev) => [
        ...prev,
        { role: 'ai', content: 'Could not reach the local agent. Check that the backend and Ollama are running.' },
      ]);
      setStreamingContent('');
    } finally {
      setLoading(false);
    }
  };

  const handleFileUpload = async (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file) return;

    setUploading(true);
    setUploadNotice('');

    const formData = new FormData();
    formData.append('file', file);

    try {
      const token = localStorage.getItem('access_token');
      const res = await fetch(`${BACKEND_URL}/documents/upload/${threadId}`, {
        method: 'POST',
        headers: { Authorization: `Bearer ${token}` },
        body: formData,
      });

      if (res.ok) {
        setUploadNotice(`Indexed "${file.name}". The agent can now reference it.`);
        loadDocuments();
        setTimeout(() => setUploadNotice(''), 4000);
      } else {
        alert('File upload failed.');
      }
    } catch (err) {
      console.error(err);
    } finally {
      setUploading(false);
      if (fileInputRef.current) fileInputRef.current.value = '';
    }
  };

  return (
    <div className="flex-1 flex flex-col h-full relative overflow-hidden bg-[var(--background)]">
      {/* Header */}
      <div className="h-16 px-5 border-b border-[var(--border)] bg-[var(--surface)] flex justify-between items-center shrink-0">
        <div className="min-w-0">
          <h2 className="text-sm font-semibold flex items-center gap-2">
            <Cpu className="w-4 h-4 text-[var(--accent)]" />
            Memory agent
          </h2>
          <p className="text-xs text-[var(--subtle)] truncate font-mono">Thread {threadId.slice(0, 8)}</p>
        </div>
        <div className="flex items-center gap-3">
          <input type="file" ref={fileInputRef} onChange={handleFileUpload} className="hidden" accept=".pdf,.docx,.txt" />
          <button onClick={() => fileInputRef.current?.click()} disabled={uploading} className="btn btn-secondary px-3 py-2 text-sm">
            {uploading ? <Loader className="w-4 h-4 animate-spin" /> : <Upload className="w-4 h-4" />}
            Upload doc
          </button>
        </div>
      </div>

      {/* Uploaded documents banner */}
      {uploadedDocs.length > 0 && (
        <div className="px-5 py-2 bg-[var(--surface-alt)] border-b border-[var(--border)] flex items-center gap-3 overflow-x-auto scrollbar-none shrink-0">
          <span className="text-[10px] uppercase font-bold tracking-wider text-[var(--muted)] shrink-0">Grounding Docs:</span>
          <div className="flex items-center gap-2">
            {uploadedDocs.map((doc: any) => (
              <div key={doc.id} className="px-2.5 py-1 bg-slate-900/60 border border-slate-800 text-[var(--foreground)] text-[10px] rounded-lg font-medium flex items-center gap-1.5 shadow-sm">
                <span className="max-w-[150px] truncate">{doc.filename}</span>
                <span className="text-[8px] text-[var(--muted)] font-mono">({(doc.size_bytes / 1024).toFixed(1)} KB)</span>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Upload notice */}
      {uploadNotice && (
        <div className="absolute top-20 right-4 bg-[var(--success-soft)] border border-[var(--success-border)] text-[var(--success)] px-4 py-3 rounded-xl text-sm flex items-center gap-2 z-20 animate-fade-in shadow-md">
          <Check className="w-4 h-4" />
          <span>{uploadNotice}</span>
        </div>
      )}

      {/* Messages */}
      <div className="flex-1 overflow-y-auto p-6 flex flex-col gap-5">
        {messages.length === 0 && !loading && (
          <div className="flex flex-col items-center justify-center h-full text-center gap-4 max-w-md mx-auto">
            <div className="p-4 bg-[var(--accent-soft)] border border-[var(--accent-border)] rounded-2xl">
              <BrainCircuit className="w-10 h-10 text-[var(--accent)]" />
            </div>
            <div>
              <h3 className="font-semibold text-base">Start the conversation</h3>
              <p className="text-sm text-[var(--muted)] mt-1.5 leading-relaxed">
                Tell the agent something about yourself or ask a question. It saves useful facts and brings them
                back in later replies.
              </p>
            </div>
          </div>
        )}

        {messages.map((m, idx) => (
          <div key={idx} className={`flex gap-3 max-w-2xl ${m.role === 'human' ? 'self-end flex-row-reverse' : 'self-start'}`}>
            <div
              className={`p-2 rounded-lg shrink-0 flex items-center justify-center h-fit border ${
                m.role === 'human'
                  ? 'bg-[var(--accent)] border-transparent'
                  : 'bg-[var(--surface)] border-[var(--border)]'
              }`}
            >
              {m.role === 'human' ? <User className="w-4 h-4 text-white" /> : <Cpu className="w-4 h-4 text-[var(--accent)]" />}
            </div>
            <div
              className={`px-4 py-3 rounded-2xl text-sm leading-relaxed border ${
                m.role === 'human'
                  ? 'bg-[var(--accent)] text-white border-transparent rounded-tr-sm'
                  : 'bg-[var(--surface)] text-[var(--foreground)] border-[var(--border)] rounded-tl-sm shadow-sm'
              }`}
            >
              <div className="whitespace-pre-wrap">{m.content}</div>
            </div>
          </div>
        ))}

        {/* Tool log */}
        {toolLogs.length > 0 && (
          <div className="card p-4 flex flex-col gap-1.5 max-w-2xl self-start font-mono text-xs text-[var(--muted)]">
            <div className="flex items-center gap-2 pb-1.5 mb-1 border-b border-[var(--border)] text-[var(--subtle)] uppercase tracking-wider font-semibold text-[10px]">
              <Terminal className="w-3.5 h-3.5" /> Tool activity
            </div>
            {toolLogs.map((log, idx) => (
              <div key={idx}>{log}</div>
            ))}
          </div>
        )}

        {/* Streaming reply */}
        {streamingContent && (
          <div className="flex gap-3 max-w-2xl self-start">
            <div className="p-2 rounded-lg shrink-0 flex items-center justify-center bg-[var(--surface)] border border-[var(--border)] h-fit">
              <Cpu className="w-4 h-4 text-[var(--accent)] animate-pulse" />
            </div>
            <div className="px-4 py-3 rounded-2xl rounded-tl-sm text-sm leading-relaxed bg-[var(--surface)] border border-[var(--border)] shadow-sm">
              <div className="whitespace-pre-wrap">
                {streamingContent}
                <span className="inline-block w-1.5 h-4 bg-[var(--accent)] ml-0.5 align-middle animate-blink" />
              </div>
            </div>
          </div>
        )}

        <div ref={messagesEndRef} />
      </div>

      {/* Input */}
      <div className="p-4 border-t border-[var(--border)] bg-[var(--surface)] shrink-0">
        <form onSubmit={handleSend} className="max-w-3xl mx-auto flex gap-2.5">
          <input
            type="text"
            required
            value={input}
            onChange={(e) => setInput(e.target.value)}
            disabled={loading}
            placeholder="Send a message…"
            className="input flex-1"
          />
          <button type="submit" disabled={loading} className="btn btn-primary px-4">
            {loading ? <Loader className="w-5 h-5 animate-spin" /> : <Send className="w-5 h-5" />}
          </button>
        </form>
      </div>
    </div>
  );
}
