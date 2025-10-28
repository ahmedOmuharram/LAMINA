import { useState } from 'react';
import { Badge } from '@/components/ui/badge';
import { getToolActualStatus } from '@/lib/toolUtils';

interface ToolCallDisplayProps {
  toolName: string;
  duration?: number;
  status?: 'started' | 'completed' | 'error';
  output?: any;
}

export function ToolCallDisplay({ toolName, duration, status = 'started', output }: ToolCallDisplayProps) {
  const [isExpanded, setIsExpanded] = useState(false);

  const formatOutput = (data: any): string => {
    if (!data) return '';
    if (typeof data === 'string') return data;
    try {
      return JSON.stringify(data, null, 2);
    } catch {
      return String(data);
    }
  };

  const actualStatus = getToolActualStatus(status, output);
  const formattedOutput = formatOutput(output);

  return (
    <div className="backdrop-blur-xl bg-gradient-to-br from-purple-100/40 to-blue-100/40 border border-purple-200/50 rounded-2xl overflow-hidden shadow-lg">
      <button
        onClick={() => setIsExpanded(!isExpanded)}
        className="w-full flex items-center justify-between p-4 hover:bg-white/20 transition-colors"
      >
        <div className="flex items-center gap-3">
          <div className="h-10 w-10 rounded-xl bg-gradient-to-br from-[#0b63c1] to-[#47b9ff] flex items-center justify-center shadow-md">
            {actualStatus === 'started' ? (
              <svg className="w-5 h-5 text-white animate-spin" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 4v5h.582m15.356 2A8.001 8.001 0 004.582 9m0 0H9m11 11v-5h-.581m0 0a8.003 8.003 0 01-15.357-2m15.357 2H15" />
              </svg>
            ) : actualStatus === 'error' ? (
              <svg className="w-5 h-5 text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
              </svg>
            ) : (
              <svg className="w-5 h-5 text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M5 13l4 4L19 7" />
              </svg>
            )}
          </div>
          <div className="flex flex-col items-start">
            <span className="font-medium text-gray-800 text-sm">{toolName}</span>
            <div className="flex items-center gap-2 mt-1">
              {duration !== undefined && duration > 0 && (
                <span className="text-xs text-gray-600">{duration.toFixed(2)}s</span>
              )}
              <Badge 
                variant={
                  actualStatus === 'completed' ? 'default' : 
                  actualStatus === 'error' ? 'destructive' : 
                  'secondary'
                }
                className="text-xs h-5"
              >
                {actualStatus}
              </Badge>
            </div>
          </div>
        </div>
        {output && (
          <svg
            className={`h-5 w-5 text-gray-600 transition-transform ${isExpanded ? 'rotate-180' : ''}`}
            xmlns="http://www.w3.org/2000/svg"
            viewBox="0 0 20 20"
            fill="currentColor"
          >
            <path
              fillRule="evenodd"
              d="M5.293 7.293a1 1 0 011.414 0L10 10.586l3.293-3.293a1 1 0 111.414 1.414l-4 4a1 1 0 01-1.414 0l-4-4a1 1 0 010-1.414z"
              clipRule="evenodd"
            />
          </svg>
        )}
      </button>

      {isExpanded && output && (
        <div className="border-t border-gray-200/50 p-4 bg-white/20">
          <pre className="text-xs text-gray-700 whitespace-pre-wrap font-mono overflow-x-auto">
            {formattedOutput}
          </pre>
        </div>
      )}
    </div>
  );
}
