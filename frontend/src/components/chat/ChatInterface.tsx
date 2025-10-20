import { useEffect, useRef } from 'react';
import { ScrollArea } from '@/components/ui/scroll-area';
import { Button } from '@/components/ui/button';
import { ChatMessage } from './ChatMessage';
import { ChatInput } from './ChatInput';

interface ChatInterfaceProps {
  chatData: ReturnType<typeof import('@/hooks/useChat').useChat>;
  onOpenAnalytics: () => void;
}

export function ChatInterface({ chatData, onOpenAnalytics }: ChatInterfaceProps) {
  const { messages, isLoading, streamingState, sendMessage, clearMessages, stopGeneration } = chatData;

  const scrollAreaRef = useRef<HTMLDivElement>(null);
  const hasStreamingContent = streamingState.text || streamingState.toolCalls.length > 0 || 
                              streamingState.images.length > 0 || streamingState.analyses.length > 0;

  // Auto-scroll to bottom when content updates
  useEffect(() => {
    const timer = setTimeout(() => {
      if (scrollAreaRef.current) {
        const scrollContainer = scrollAreaRef.current.querySelector('[data-radix-scroll-area-viewport]');
        if (scrollContainer) {
          scrollContainer.scrollTo({
            top: scrollContainer.scrollHeight,
            behavior: 'smooth'
          });
        }
      }
    }, 100);
    return () => clearTimeout(timer);
  }, [messages, streamingState, isLoading]);

  return (
    <div className="flex flex-col h-full">
      {/* Header */}
      <div className="flex items-center justify-between px-6 py-3 border-b border-white/10 backdrop-blur-2xl bg-white/5 shadow-sm">
        <Button 
          variant="ghost" 
          size="sm" 
          onClick={clearMessages}
          className="h-8 text-gray-600 hover:text-gray-900 hover:bg-gray-100/50"
        >
          <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 7l-.867 12.142A2 2 0 0116.138 21H7.862a2 2 0 01-1.995-1.858L5 7m5 4v6m4-6v6m1-10V4a1 1 0 00-1-1h-4a1 1 0 00-1 1v3M4 7h16" />
          </svg>
        </Button>
        
        <Button 
          variant="ghost" 
          size="sm" 
          onClick={onOpenAnalytics}
          className="h-8 text-gray-600 hover:text-gray-900 hover:bg-gray-100/50 flex items-center gap-2"
        >
          <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 19v-6a2 2 0 00-2-2H5a2 2 0 00-2 2v6a2 2 0 002 2h2a2 2 0 002-2zm0 0V9a2 2 0 012-2h2a2 2 0 012 2v10m-6 0a2 2 0 002 2h2a2 2 0 002-2m0 0V5a2 2 0 012-2h2a2 2 0 012 2v14a2 2 0 01-2 2h-2a2 2 0 01-2-2z" />
          </svg>
          <span className="text-sm">Analytics</span>
        </Button>
      </div>

      {/* Messages */}
      <div className="flex-1 overflow-hidden">
        <ScrollArea className="h-full p-6" ref={scrollAreaRef}>
          <div className="max-w-4xl mx-auto space-y-6 pb-8">
          {messages.length === 0 && !hasStreamingContent && (
            <div className="backdrop-blur-xl bg-white/15 border border-white/20 rounded-3xl p-10 text-center shadow-xl">
              <div className="max-w-md mx-auto">
                <div className="w-14 h-14 mx-auto mb-3 rounded-2xl bg-gradient-to-br from-[#0b63c1] to-[#47b9ff] flex items-center justify-center shadow-lg">
                  <svg className="w-7 h-7 text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 10V3L4 14h7v7l9-11h-7z" />
                  </svg>
                </div>
                <h3 className="text-xl font-bold text-gray-800 mb-1.5">Materials Science AI</h3>
                <p className="text-gray-600 mb-6 text-xs">
                  Ask me about phase diagrams, battery calculations, or materials properties
                </p>
                <div className="grid grid-cols-1 gap-2 text-xs text-left">
                  <div className="p-3 backdrop-blur-xl bg-white/15 border border-white/20 rounded-2xl hover:bg-white/25 transition-colors">
                    <div className="flex items-start gap-2">
                      <span className="text-lg">💎</span>
                      <div>
                        <strong className="text-gray-800 block mb-0.5 text-xs">CALPHAD</strong>
                        <span className="text-gray-600 text-[10px]">Phase diagrams, thermodynamic calculations</span>
                      </div>
                    </div>
                  </div>
                  <div className="p-3 backdrop-blur-xl bg-white/15 border border-white/20 rounded-2xl hover:bg-white/25 transition-colors">
                    <div className="flex items-start gap-2">
                      <span className="text-lg">🔋</span>
                      <div>
                        <strong className="text-gray-800 block mb-0.5 text-xs">Electrochemistry</strong>
                        <span className="text-gray-600 text-[10px]">Battery analysis, voltage profiles</span>
                      </div>
                    </div>
                  </div>
                  <div className="p-3 backdrop-blur-xl bg-white/15 border border-white/20 rounded-2xl hover:bg-white/25 transition-colors">
                    <div className="flex items-start gap-2">
                      <span className="text-lg">🔍</span>
                      <div>
                        <strong className="text-gray-800 block mb-0.5 text-xs">Materials Search</strong>
                        <span className="text-gray-600 text-[10px]">Properties, structures, compositions</span>
                      </div>
                    </div>
                  </div>
                </div>
              </div>
            </div>
          )}

          {messages.map((msg, idx) => (
            <ChatMessage key={idx} message={msg} />
          ))}

          {hasStreamingContent && (
            <ChatMessage
              message={{
                role: 'assistant',
                content: streamingState.text,
              }}
              streamingImages={streamingState.images}
              streamingAnalyses={streamingState.analyses}
            />
          )}

          {isLoading && !hasStreamingContent && (
            <div className="flex gap-3">
              <div className="h-10 w-10 rounded-full bg-white border border-gray-200 flex items-center justify-center shadow-lg">
                <img src="/gpt.svg" alt="GPT" className="w-6 h-6" />
              </div>
              <div className="backdrop-blur-xl bg-white/15 border border-white/20 rounded-3xl rounded-tl-sm p-4 shadow-xl">
                <div className="flex gap-2">
                  <div className="h-2 w-2 rounded-full bg-gray-400 animate-bounce" style={{ animationDelay: '0ms' }} />
                  <div className="h-2 w-2 rounded-full bg-gray-400 animate-bounce" style={{ animationDelay: '150ms' }} />
                  <div className="h-2 w-2 rounded-full bg-gray-400 animate-bounce" style={{ animationDelay: '300ms' }} />
                </div>
              </div>
            </div>
          )}
          </div>
        </ScrollArea>
      </div>

      {/* Input */}
      <div className="p-4 border-t border-white/10 backdrop-blur-2xl bg-white/5 shadow-sm">
        <div className="max-w-4xl mx-auto">
          <ChatInput
            onSend={sendMessage}
            disabled={isLoading}
            onStop={stopGeneration}
            isGenerating={isLoading}
          />
        </div>
      </div>
    </div>
  );
}
