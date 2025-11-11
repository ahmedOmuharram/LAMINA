import { useState, useCallback, useRef } from 'react';
import { apiClient } from '@/lib/api';
import type { ChatMessage, StreamChunk, ConversationMetrics, ToolCall, ImageData, AnalysisData } from '@/types/api';

export interface UseChatOptions {
  model?: string;
  onError?: (error: Error) => void;
  enabledFunctions?: string[];
}

interface StreamingState {
  text: string;
  toolCalls: ToolCall[];
  images: ImageData[];
  analyses: AnalysisData[];
}

export function useChat(options: UseChatOptions = {}) {
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [isLoading, setIsLoading] = useState(false);
  const [streamingState, setStreamingState] = useState<StreamingState>({
    text: '',
    toolCalls: [],
    images: [],
    analyses: [],
  });
  const [metrics, setMetrics] = useState<ConversationMetrics>({
    messageCount: 0,
    totalTokens: 0,
    totalDuration: 0,
    toolCalls: [],
    model: options.model || 'gpt-4o-mini',
  });

  const abortControllerRef = useRef<AbortController | null>(null);
  const startTimeRef = useRef<number>(0);

  const sendMessage = useCallback(async (content: string | ChatMessage['content'], images?: File[]) => {
    if (isLoading) return;

    setIsLoading(true);
    setStreamingState({ text: '', toolCalls: [], images: [], analyses: [] });
    startTimeRef.current = Date.now();

    // Build the user message
    let messageContent: ChatMessage['content'] = content;
    
    if (images && images.length > 0) {
      const imageUrls = await Promise.all(
        images.map(img => apiClient.uploadImage(img))
      );
      
      const textContent = typeof content === 'string' ? content : '';
      messageContent = [
        { type: 'text', text: textContent },
        ...imageUrls.map(url => ({
          type: 'image_url' as const,
          image_url: { url, detail: 'high' as const },
        })),
      ];
    }

    const userMessage: ChatMessage = {
      role: 'user',
      content: messageContent,
    };

    const newMessages = [...messages, userMessage];
    setMessages(newMessages);

    try {
      abortControllerRef.current = new AbortController();
      const streamState: StreamingState = {
        text: '',
        toolCalls: [],
        images: [],
        analyses: [],
      };
      let tokenCount = 0;

      // Stream the response
      for await (const chunk of apiClient.streamMessage({
        messages: newMessages,
        model: options.model || 'gpt-4o-mini',
        stream: true,
        enabled_functions: options.enabledFunctions,
      })) {
        try {
          const parsed: StreamChunk = JSON.parse(chunk);
          const delta = parsed.choices[0]?.delta;
          
          if (delta?.content) {
            streamState.text += delta.content;
          }
          
          // Handle usage information from backend (accurate tiktoken count)
          if (delta?.usage) {
            tokenCount = delta.usage.total_tokens;
          }

          // Handle tool call events
          if (delta?.tool_call) {
            const toolCall = delta.tool_call;
            const existingIndex = streamState.toolCalls.findIndex(t => t.id === toolCall.id);
            
            if (toolCall.status === 'started') {
              if (existingIndex === -1) {
                streamState.toolCalls.push({
                  id: toolCall.id,
                  name: toolCall.name,
                  timestamp: Date.now(),
                  duration: 0,
                  status: 'started',
                  input: toolCall.input,
                });
              }
            } else if (toolCall.status === 'completed') {
              if (existingIndex !== -1) {
                streamState.toolCalls[existingIndex] = {
                  ...streamState.toolCalls[existingIndex],
                  duration: toolCall.duration || 0,
                  status: 'completed',
                  input: toolCall.input || streamState.toolCalls[existingIndex].input,
                  output: toolCall.output,
                };
              } else {
                // Tool wasn't registered as started, add it directly
                streamState.toolCalls.push({
                  id: toolCall.id,
                  name: toolCall.name,
                  timestamp: Date.now(),
                  duration: toolCall.duration || 0,
                  status: 'completed',
                  input: toolCall.input,
                  output: toolCall.output,
                });
              }
            }
          }

          // Handle image events
          if (delta?.image) {
            streamState.images.push(delta.image);
          }

          // Handle analysis events
          if (delta?.analysis) {
            streamState.analyses.push(delta.analysis);
          }

          // Update the streaming state for display
          setStreamingState({ ...streamState });

          if (parsed.choices[0]?.finish_reason === 'stop') {
            break;
          }
        } catch (parseError) {
          console.warn('Failed to parse chunk:', chunk, parseError);
        }
      }

      const duration = Date.now() - startTimeRef.current;
      
      // Build the assistant message preserving all structured data
      const assistantMessage: ChatMessage = {
        role: 'assistant',
        content: streamState.text || '',
        toolCalls: streamState.toolCalls.length > 0 ? streamState.toolCalls : undefined,
        images: streamState.images.length > 0 ? streamState.images : undefined,
        analyses: streamState.analyses.length > 0 ? streamState.analyses : undefined,
      };

      setMessages([...newMessages, assistantMessage]);
      setStreamingState({ text: '', toolCalls: [], images: [], analyses: [] });

      // Update metrics
      setMetrics(prev => ({
        ...prev,
        messageCount: prev.messageCount + 2, // user + assistant
        totalTokens: prev.totalTokens + tokenCount,
        totalDuration: prev.totalDuration + duration,
        toolCalls: [...prev.toolCalls, ...streamState.toolCalls],
      }));
    } catch (error) {
      console.error('Error sending message:', error);
      options.onError?.(error as Error);
    } finally {
      setIsLoading(false);
      abortControllerRef.current = null;
    }
  }, [messages, isLoading, options]);

  const clearMessages = useCallback(() => {
    setMessages([]);
    setStreamingState({ text: '', toolCalls: [], images: [], analyses: [] });
    setMetrics({
      messageCount: 0,
      totalTokens: 0,
      totalDuration: 0,
      toolCalls: [],
      model: options.model || 'gpt-4o-mini',
    });
  }, [options.model]);

  const stopGeneration = useCallback(() => {
    if (abortControllerRef.current) {
      abortControllerRef.current.abort();
      setIsLoading(false);
      setStreamingState({ text: '', toolCalls: [], images: [], analyses: [] });
    }
  }, []);

  return {
    messages,
    isLoading,
    streamingState,
    metrics,
    sendMessage,
    clearMessages,
    stopGeneration,
  };
}
