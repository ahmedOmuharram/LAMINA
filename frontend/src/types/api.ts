// API types matching the backend schema

export interface ChatMessage {
  role: 'user' | 'assistant' | 'system';
  content: string | MessageContent[];
  toolCalls?: ToolCall[];
  images?: ImageData[];
  analyses?: AnalysisData[];
}

export type MessageContent = TextContent | ImageContent;

export interface TextContent {
  type: 'text';
  text: string;
}

export interface ImageContent {
  type: 'image_url';
  image_url: {
    url: string;
    detail?: 'low' | 'high' | 'auto';
  };
}

export interface ChatRequest {
  messages: ChatMessage[];
  model?: string;
  stream?: boolean;
  temperature?: number;
  max_tokens?: number;
}

export interface ChatResponse {
  id: string;
  object: string;
  created: number;
  model: string;
  choices: Array<{
    index: number;
    message: {
      role: string;
      content: string;
    };
    finish_reason: string;
  }>;
  usage: {
    prompt_tokens: number;
    completion_tokens: number;
    total_tokens: number;
  };
}

export interface ToolCallDelta {
  id: string;
  name: string;
  status: 'started' | 'completed';
  duration?: number;
  input?: any;
  output?: any;
}

export interface ImageDelta {
  url: string;
  metadata: Record<string, any>;
}

export interface AnalysisDelta {
  content: string;
}

export interface UsageDelta {
  prompt_tokens: number;
  completion_tokens: number;
  total_tokens: number;
}

export interface StreamChunk {
  id: string;
  object: string;
  created: number;
  model: string;
  choices: Array<{
    index: number;
    delta: {
      role?: string;
      content?: string;
      tool_call?: ToolCallDelta;
      image?: ImageDelta;
      analysis?: AnalysisDelta;
      usage?: UsageDelta;
    };
    finish_reason: string | null;
  }>;
}

export interface Model {
  id: string;
  object: string;
  created: number;
  owned_by: string;
  name: string;
  description: string;
}

export interface ModelsResponse {
  object: string;
  data: Model[];
}

// Analytics types
export interface ToolCall {
  id: string;
  name: string;
  timestamp: number;
  duration: number;
  status: 'started' | 'completed' | 'error';
  input?: any;
  output?: any;
}

export interface ImageData {
  url: string;
  metadata: Record<string, any>;
}

export interface AnalysisData {
  content: string;
}

export interface ConversationMetrics {
  messageCount: number;
  totalTokens: number;
  totalDuration: number;
  toolCalls: ToolCall[];
  model: string;
}

// Testing types
export interface TestQuestion {
  id: string;
  question: string;
  answer?: string;
  notes?: string;
  timestamp?: string;
  duration?: number;
  error?: string;
  toolCalls?: ToolCall[];
}

export interface TestRun {
  id: string;
  name: string;
  prompt: string;
  questions: TestQuestion[];
  model: string;
  createdAt: string;
  completedAt?: string;
  status: 'draft' | 'running' | 'completed' | 'error';
}

export interface TestTemplate {
  id: string;
  name: string;
  prompt: string;
  questions: string[];
  createdAt: string;
  updatedAt?: string;
}

