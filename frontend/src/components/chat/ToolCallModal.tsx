import {
  Dialog,
  DialogContent,
  DialogHeader,
  DialogTitle,
} from '@/components/ui/dialog';
import { Badge } from '@/components/ui/badge';
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

  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      <DialogContent className="max-w-3xl max-h-[80vh] overflow-hidden flex flex-col">
        <DialogHeader>
          <div className="flex items-center gap-3">
            <DialogTitle className="text-xl">{toolCall.name}</DialogTitle>
            <Badge 
              variant={
                toolCall.status === 'completed' ? 'default' : 
                toolCall.status === 'error' ? 'destructive' : 
                'secondary'
              }
            >
              {toolCall.status}
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

          {/* Output */}
          {toolCall.output && (
            <div>
              <h3 className="text-sm font-semibold text-gray-700 mb-2">Output</h3>
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

