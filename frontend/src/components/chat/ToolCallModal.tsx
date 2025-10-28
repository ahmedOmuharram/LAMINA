import {
  Dialog,
  DialogContent,
  DialogHeader,
  DialogTitle,
} from '@/components/ui/dialog';
import { Badge } from '@/components/ui/badge';
import { parseToolOutput, getToolActualStatus } from '@/lib/toolUtils';
import type { ToolCall } from '@/types/api';

interface ToolCallModalProps {
  toolCall: ToolCall | null;
  open: boolean;
  onOpenChange: (open: boolean) => void;
}

export function ToolCallModal({ toolCall, open, onOpenChange }: ToolCallModalProps) {
  if (!toolCall) return null;

  const formatJSON = (data: any): string => {
    if (!data) return 'N/A';
    if (typeof data === 'string') {
      try {
        return JSON.stringify(JSON.parse(data), null, 2);
      } catch {
        return data;
      }
    }
    return JSON.stringify(data, null, 2);
  };

  const outputInfo = parseToolOutput(toolCall.output);
  const actualStatus = getToolActualStatus(toolCall.status, toolCall.output);

  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      <DialogContent className="max-w-3xl max-h-[80vh] overflow-hidden flex flex-col">
        <DialogHeader>
          <div className="flex items-center gap-3">
            <DialogTitle className="text-xl">{toolCall.name}</DialogTitle>
            <Badge 
              variant={
                actualStatus === 'completed' ? 'default' : 
                actualStatus === 'error' ? 'destructive' : 
                'secondary'
              }
            >
              {actualStatus}
            </Badge>
            {toolCall.duration > 0 && (
              <span className="text-sm text-gray-500">
                {toolCall.duration.toFixed(2)}s
              </span>
            )}
          </div>
        </DialogHeader>

        <div className="flex-1 overflow-y-auto space-y-4 pr-2">
          {/* Input Parameters */}
          <div>
            <h3 className="text-sm font-semibold text-gray-700 mb-2">Input Parameters</h3>
            <div className="bg-gray-50 rounded-lg p-4 border border-gray-200">
              <pre className="text-xs font-mono overflow-x-auto whitespace-pre-wrap">
                {formatJSON(toolCall.input)}
              </pre>
            </div>
          </div>

          {/* Error Message (if standardized result wrapper with error) */}
          {outputInfo.hasWrapper && !outputInfo.success && outputInfo.error && (
            <div>
              <h3 className="text-sm font-semibold text-red-700 mb-2">Error</h3>
              <div className="bg-red-50 rounded-lg p-4 border border-red-200">
                <div className="flex items-start gap-2">
                  <svg className="w-5 h-5 text-red-600 flex-shrink-0 mt-0.5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 8v4m0 4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
                  </svg>
                  <div className="flex-1">
                    <p className="text-sm text-red-800 font-medium">{outputInfo.error}</p>
                    {outputInfo.errorType && (
                      <p className="text-xs text-red-600 mt-1">Type: {outputInfo.errorType}</p>
                    )}
                  </div>
                </div>
              </div>
            </div>
          )}

          {/* Output */}
          {toolCall.output && (
            <div>
              <h3 className="text-sm font-semibold text-gray-700 mb-2">
                {outputInfo.hasWrapper ? 'Full Response' : 'Output'}
              </h3>
              <div className="bg-gray-50 rounded-lg p-4 border border-gray-200">
                <pre className="text-xs font-mono overflow-x-auto whitespace-pre-wrap">
                  {formatJSON(toolCall.output)}
                </pre>
              </div>
            </div>
          )}
        </div>
      </DialogContent>
    </Dialog>
  );
}

