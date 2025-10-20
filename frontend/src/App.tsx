import { useState, useEffect } from 'react';
import { ChatInterface } from '@/components/chat/ChatInterface';
import { ModelSelector } from '@/components/layout/ModelSelector';
import { AnalyticsModal } from '@/components/analytics/AnalyticsModal';
import { ToolCallModal } from '@/components/chat/ToolCallModal';
import { TestingInterface } from '@/components/testing/TestingInterface';
import { useChat } from '@/hooks/useChat';
import type { ToolCall } from '@/types/api';

// Component to show live elapsed time for running tools
function ToolDuration({ tool }: { tool: ToolCall }) {
  const [elapsed, setElapsed] = useState(0);

  useEffect(() => {
    if (tool.status === 'started') {
      const startTime = new Date(tool.timestamp).getTime();
      
      const updateElapsed = () => {
        const now = Date.now();
        const elapsedSeconds = (now - startTime) / 1000;
        setElapsed(elapsedSeconds);
      };
      
      // Update immediately
      updateElapsed();
      
      // Then update every 100ms for smooth counting
      const interval = setInterval(updateElapsed, 100);
      
      return () => clearInterval(interval);
    } else if (tool.duration && tool.duration > 0) {
      setElapsed(tool.duration);
    }
  }, [tool.status, tool.timestamp, tool.duration]);

  if (elapsed > 0) {
    return (
      <div className="text-[9px] text-gray-500">
        {elapsed.toFixed(1)}s
      </div>
    );
  }
  
  return null;
}

function App() {
  const [selectedModel, setSelectedModel] = useState('gpt-4o-mini');
  const [selectedToolCall, setSelectedToolCall] = useState<ToolCall | null>(null);
  const [isToolModalOpen, setIsToolModalOpen] = useState(false);
  const [isAnalyticsOpen, setIsAnalyticsOpen] = useState(false);
  const [activeView, setActiveView] = useState<'chat' | 'testing'>('chat');
  const chatData = useChat({ model: selectedModel });
  const { metrics, streamingState, isLoading } = chatData;
  
  const handleToolClick = (tool: ToolCall) => {
    setSelectedToolCall(tool);
    setIsToolModalOpen(true);
  };

  return (
    <div className="flex h-screen bg-gradient-to-br from-blue-50 via-purple-50 to-pink-50">
      {/* Sidebar */}
      <aside className="w-20 border-r border-gray-200/30 backdrop-blur-2xl bg-white/20 flex flex-col items-center py-6 gap-6">
        {/* Logo */}
        <div 
          className="w-14 h-14 cursor-pointer hover:scale-105 transition-transform flex items-center justify-center"
          title="LAMINA: LLM-Assisted Material INformatics and Analysis"
        >
          <img src="/lamina.svg" alt="LAMINA" className="w-12 h-12" />
        </div>

        {/* Navigation Icons */}
        <div className="flex flex-col gap-3">
          <button
            onClick={() => setActiveView('chat')}
            className={`w-12 h-12 rounded-xl flex items-center justify-center transition-all ${
              activeView === 'chat'
                ? 'bg-gradient-to-br from-[#0b63c1] to-[#47b9ff] text-white shadow-lg'
                : 'bg-white/40 text-gray-600 hover:bg-white/60'
            }`}
            title="Chat"
          >
            <svg className="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M8 12h.01M12 12h.01M16 12h.01M21 12c0 4.418-4.03 8-9 8a9.863 9.863 0 01-4.255-.949L3 20l1.395-3.72C3.512 15.042 3 13.574 3 12c0-4.418 4.03-8 9-8s9 3.582 9 8z" />
            </svg>
          </button>

          <button
            onClick={() => setActiveView('testing')}
            className={`w-12 h-12 rounded-xl flex items-center justify-center transition-all ${
              activeView === 'testing'
                ? 'bg-gradient-to-br from-[#0b63c1] to-[#47b9ff] text-white shadow-lg'
                : 'bg-white/40 text-gray-600 hover:bg-white/60'
            }`}
            title="Testing"
          >
            <svg className="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 5H7a2 2 0 00-2 2v12a2 2 0 002 2h10a2 2 0 002-2V7a2 2 0 00-2-2h-2M9 5a2 2 0 002 2h2a2 2 0 002-2M9 5a2 2 0 012-2h2a2 2 0 012 2m-6 9l2 2 4-4" />
            </svg>
          </button>
        </div>

        {/* Spacer */}
        <div className="flex-1" />

        {/* Settings at bottom */}
        <button 
          className="w-14 h-14 rounded-2xl backdrop-blur-xl bg-white/20 border border-gray-200/30 flex items-center justify-center hover:bg-white/30 transition-colors"
          title="Settings"
        >
          <svg className="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M10.325 4.317c.426-1.756 2.924-1.756 3.35 0a1.724 1.724 0 002.573 1.066c1.543-.94 3.31.826 2.37 2.37a1.724 1.724 0 001.065 2.572c1.756.426 1.756 2.924 0 3.35a1.724 1.724 0 00-1.066 2.573c.94 1.543-.826 3.31-2.37 2.37a1.724 1.724 0 00-2.572 1.065c-.426 1.756-2.924 1.756-3.35 0a1.724 1.724 0 00-2.573-1.066c-1.543.94-3.31-.826-2.37-2.37a1.724 1.724 0 00-1.065-2.572c-1.756-.426-1.756-2.924 0-3.35a1.724 1.724 0 001.066-2.573c-.94-1.543.826-3.31 2.37-2.37.996.608 2.296.07 2.572-1.065z" />
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M15 12a3 3 0 11-6 0 3 3 0 016 0z" />
          </svg>
        </button>
      </aside>

      {/* Model Selector Sidebar */}
      <aside className="w-64 border-r border-white/20 backdrop-blur-2xl bg-white/10 flex flex-col shadow-lg">
        <div className="p-4 border-b border-white/10">
          <h1 className="text-xl font-bold bg-gradient-to-r from-[#0b63c1] to-[#47b9ff] bg-clip-text text-transparent">
            LAMINA
          </h1>
          <p className="text-[10px] text-gray-600 mt-0.5 leading-tight">
            LLM-Assisted Material INformatics and Analysis
          </p>
        </div>
        
        <div className="flex-1 p-4 flex flex-col gap-4 overflow-y-auto">
          <ModelSelector value={selectedModel} onChange={setSelectedModel} />
          
          {/* Active Tools (streaming) */}
          {isLoading && streamingState.toolCalls.length > 0 && (
            <div>
              <h3 className="text-[10px] font-semibold text-gray-600 uppercase tracking-wider mb-2">Active Tools</h3>
              <div className="space-y-1.5">
                {streamingState.toolCalls.map((tool) => (
                  <button
                    key={tool.id}
                    onClick={() => handleToolClick(tool)}
                    className="w-full backdrop-blur-xl bg-white/20 border border-white/20 rounded-xl p-2 shadow-sm hover:bg-white/30 transition-colors text-left"
                  >
                    <div className="flex items-center gap-1.5 mb-0.5">
                      {tool.status === 'started' && (
                        <svg className="w-2.5 h-2.5 text-blue-500 animate-spin flex-shrink-0" fill="none" viewBox="0 0 24 24">
                          <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" />
                          <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z" />
                        </svg>
                      )}
                      <span className="text-[10px] font-medium text-gray-800 truncate flex-1">{tool.name}</span>
                      <span className={`text-[9px] px-1.5 py-0.5 rounded-full flex-shrink-0 ${
                        tool.status === 'completed' ? 'bg-green-100/70 text-green-700' :
                        tool.status === 'error' ? 'bg-red-100/70 text-red-700' :
                        'bg-blue-100/70 text-blue-700'
                      }`}>
                        {tool.status}
                      </span>
                    </div>
                    <ToolDuration tool={tool} />
                  </button>
                ))}
              </div>
            </div>
          )}
          
          {/* Recent Tools */}
          {metrics.toolCalls.length > 0 && (
            <div>
              <h3 className="text-[10px] font-semibold text-gray-600 uppercase tracking-wider mb-2">Recent Tools</h3>
              <div className="space-y-1.5">
                {metrics.toolCalls.slice(-5).reverse().map((tool, idx) => (
                  <button
                    key={idx}
                    onClick={() => handleToolClick(tool)}
                    className="w-full backdrop-blur-xl bg-white/20 border border-white/20 rounded-xl p-2 shadow-sm hover:bg-white/30 transition-colors text-left"
                  >
                    <div className="flex items-center justify-between mb-0.5">
                      <span className="text-[10px] font-medium text-gray-800 truncate">{tool.name}</span>
                      <span className={`text-[9px] px-1.5 py-0.5 rounded-full flex-shrink-0 ${
                        tool.status === 'completed' ? 'bg-green-100/70 text-green-700' :
                        tool.status === 'error' ? 'bg-red-100/70 text-red-700' :
                        'bg-gray-100/70 text-gray-600'
                      }`}>
                        {tool.status}
                      </span>
                    </div>
                    <ToolDuration tool={tool} />
                  </button>
                ))}
              </div>
            </div>
          )}
        </div>
      </aside>

      {/* Main Content */}
      <main className="flex-1 flex flex-col overflow-hidden">
        {activeView === 'chat' ? (
          <ChatInterface chatData={chatData} onOpenAnalytics={() => setIsAnalyticsOpen(true)} />
        ) : (
          <TestingInterface selectedModel={selectedModel} />
        )}
      </main>

      {/* Tool Call Details Modal */}
      <ToolCallModal 
        toolCall={selectedToolCall} 
        open={isToolModalOpen} 
        onOpenChange={setIsToolModalOpen}
      />

      {/* Analytics Modal */}
      <AnalyticsModal 
        metrics={metrics}
        open={isAnalyticsOpen}
        onOpenChange={setIsAnalyticsOpen}
      />
    </div>
  );
}

export default App;
