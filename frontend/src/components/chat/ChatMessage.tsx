import { useState } from 'react';
import { Avatar, AvatarFallback } from '@/components/ui/avatar';
import { Button } from '@/components/ui/button';
import { AnalysisModal } from './AnalysisModal';
import type { ChatMessage as ChatMessageType, ImageData, AnalysisData } from '@/types/api';

interface ChatMessageProps {
  message: ChatMessageType;
  streamingImages?: ImageData[];
  streamingAnalyses?: AnalysisData[];
}

export function ChatMessage({ message, streamingImages = [], streamingAnalyses = [] }: ChatMessageProps) {
  const isUser = message.role === 'user';
  const [selectedAnalysisIdx, setSelectedAnalysisIdx] = useState<number | null>(null);

  const getTextContent = () => {
    if (typeof message.content === 'string') {
      return message.content;
    }
    return message.content
      .filter(part => part.type === 'text')
      .map(part => part.type === 'text' ? part.text : '')
      .join('');
  };

  const getImages = () => {
    if (typeof message.content === 'string') {
      return [];
    }
    return message.content
      .filter(part => part.type === 'image_url')
      .map(part => part.type === 'image_url' ? part.image_url.url : '');
  };

  const textContent = getTextContent();
  const images = getImages();
  
  // Use message's preserved data or streaming data
  const messageImages = message.images || streamingImages;
  const analyses = message.analyses || streamingAnalyses;

  return (
    <div className={`flex gap-3 ${isUser ? 'flex-row-reverse' : 'flex-row'}`}>
      <Avatar className="h-10 w-10 shrink-0 shadow-lg">
        <AvatarFallback className={`${
          isUser 
            ? 'bg-gradient-to-br from-[#0b63c1] to-[#47b9ff]' 
            : 'bg-white border border-gray-200'
        }`}>
          {isUser ? (
            <svg className="w-5 h-5 text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M16 7a4 4 0 11-8 0 4 4 0 018 0zM12 14a7 7 0 00-7 7h14a7 7 0 00-7-7z" />
            </svg>
          ) : (
            <img src="/gpt.svg" alt="GPT" className="w-6 h-6" />
          )}
        </AvatarFallback>
      </Avatar>

      <div className={`flex flex-col gap-3 max-w-[80%] ${isUser ? 'items-end' : 'items-start'}`}>
        {images.length > 0 && (
          <div className="grid grid-cols-2 gap-2">
            {images.map((url, idx) => (
              <div key={idx} className="backdrop-blur-xl bg-white/40 border border-gray-200/50 rounded-2xl overflow-hidden shadow-lg">
                <img
                  src={url}
                  alt={`Uploaded image ${idx + 1}`}
                  className="w-full h-48 object-cover"
                />
              </div>
            ))}
          </div>
        )}

        {/* Generated Images with Analysis Button - Show BEFORE text */}
        {messageImages.length > 0 && (
          <div className="w-full space-y-2">
            {messageImages.map((img, idx) => {
              const metadata = img.metadata;
              const system = metadata.system || 'Unknown';
              const description = metadata.description || 'Generated diagram';
              const hasAnalysis = idx < analyses.length;
              
              return (
                <div key={`streaming-image-${idx}`} className="backdrop-blur-xl bg-white/15 border border-white/20 rounded-2xl overflow-hidden shadow-xl">
                  <div className="p-3 border-b border-white/10 flex items-center justify-between backdrop-blur-sm">
                    <div className="flex-1">
                      <p className="text-xs font-medium text-gray-800">{description}</p>
                      <p className="text-[10px] text-gray-600 mt-0.5">{system}</p>
                    </div>
                    {hasAnalysis && (
                      <Button
                        size="sm"
                        onClick={() => setSelectedAnalysisIdx(idx)}
                        className="bg-gradient-to-br from-[#0b63c1] to-[#47b9ff] hover:to-[#47b9ff]/80 text-white"
                      >
                        <svg className="w-4 h-4 mr-1.5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 19v-6a2 2 0 00-2-2H5a2 2 0 00-2 2v6a2 2 0 002 2h2a2 2 0 002-2zm0 0V9a2 2 0 012-2h2a2 2 0 012 2v10m-6 0a2 2 0 002 2h2a2 2 0 002-2m0 0V5a2 2 0 012-2h2a2 2 0 012 2v14a2 2 0 01-2 2h-2a2 2 0 01-2-2z" />
                        </svg>
                        View Analysis
                      </Button>
                    )}
                  </div>
                  <img
                    src={img.url}
                    alt={`${system} ${description}`}
                    className="w-full"
                  />
                </div>
              );
            })}
          </div>
        )}

        {/* Text content with markdown - Show AFTER images */}
        {textContent && (
          <div 
            className={`backdrop-blur-2xl bg-white/15 border border-white/20 rounded-3xl p-6 shadow-xl ${
              isUser ? 'rounded-tr-sm' : 'rounded-tl-sm'
            }`}
          >
            <div className="prose prose-sm max-w-none text-gray-800 prose-headings:text-gray-900 prose-p:text-gray-800 prose-strong:text-gray-900 prose-code:text-gray-800 prose-pre:bg-gray-100/50 text-xs leading-relaxed">
              {textContent.split('\n').map((line, lineIdx) => {
                // Simple markdown-like formatting
                let formattedLine = line;
                
                // Bold
                formattedLine = formattedLine.replace(/\*\*(.+?)\*\*/g, '<strong>$1</strong>');
                
                // Italic
                formattedLine = formattedLine.replace(/\*(.+?)\*/g, '<em>$1</em>');
                
                // Code
                formattedLine = formattedLine.replace(/`([^`]+)`/g, '<code class="px-1 py-0.5 bg-gray-200 rounded text-sm">$1</code>');
                
                // Links
                formattedLine = formattedLine.replace(/\[([^\]]+)\]\(([^)]+)\)/g, '<a href="$2" class="text-blue-600 hover:underline" target="_blank" rel="noopener noreferrer">$1</a>');
                
                return (
                  <p 
                    key={lineIdx} 
                    className="mb-3 last:mb-0 text-xs"
                    dangerouslySetInnerHTML={{ __html: formattedLine || '\u00A0' }}
                  />
                );
              })}
            </div>
          </div>
        )}

        {/* Analysis Modal */}
        {selectedAnalysisIdx !== null && analyses[selectedAnalysisIdx] && (
          <AnalysisModal
            analysis={analyses[selectedAnalysisIdx].content}
            title={messageImages[selectedAnalysisIdx]?.metadata?.description || 'Phase Diagram Analysis'}
            open={selectedAnalysisIdx !== null}
            onOpenChange={(open) => !open && setSelectedAnalysisIdx(null)}
          />
        )}
      </div>
    </div>
  );
}
