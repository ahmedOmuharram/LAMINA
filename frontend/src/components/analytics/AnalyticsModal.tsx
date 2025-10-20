import {
  Dialog,
  DialogContent,
  DialogHeader,
  DialogTitle,
} from '@/components/ui/dialog';
import { Badge } from '@/components/ui/badge';
import type { ConversationMetrics } from '@/types/api';

interface AnalyticsModalProps {
  metrics: ConversationMetrics;
  open: boolean;
  onOpenChange: (open: boolean) => void;
}

export function AnalyticsModal({ metrics, open, onOpenChange }: AnalyticsModalProps) {
  // Calculate average response time (totalDuration is in milliseconds)
  // messageCount / 2 gives us number of exchanges (user + assistant pairs)
  const numExchanges = Math.max(1, metrics.messageCount / 2);
  const avgDuration = metrics.messageCount > 0 
    ? ((metrics.totalDuration / 1000) / numExchanges).toFixed(2)
    : '0';

  const successfulTools = metrics.toolCalls.filter(t => t.status === 'completed').length;
  const failedTools = metrics.toolCalls.filter(t => t.status === 'error').length;

  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      <DialogContent className="max-w-3xl max-h-[80vh] overflow-hidden flex flex-col">
        <DialogHeader>
          <DialogTitle className="text-xl">Conversation Analytics</DialogTitle>
        </DialogHeader>

        <div className="flex-1 overflow-y-auto pr-2">
          <div className="prose prose-sm max-w-none">
            {/* Statistics Grid */}
            <div className="grid grid-cols-2 gap-3 mb-4 not-prose">
              <div className="backdrop-blur-xl bg-gradient-to-br from-blue-50 to-blue-100 border border-blue-200 rounded-xl p-4">
                <div className="text-xs text-blue-600 uppercase tracking-wider font-medium">Messages</div>
                <div className="text-3xl font-bold text-blue-900 mt-1">{metrics.messageCount}</div>
              </div>
              
              <div className="backdrop-blur-xl bg-gradient-to-br from-purple-50 to-purple-100 border border-purple-200 rounded-xl p-4">
                <div className="text-xs text-purple-600 uppercase tracking-wider font-medium">Tokens</div>
                <div className="text-3xl font-bold text-purple-900 mt-1">{metrics.totalTokens}</div>
              </div>
              
              <div className="backdrop-blur-xl bg-gradient-to-br from-green-50 to-green-100 border border-green-200 rounded-xl p-4">
                <div className="text-xs text-green-600 uppercase tracking-wider font-medium">Avg Response</div>
                <div className="text-3xl font-bold text-green-900 mt-1">{avgDuration}s</div>
              </div>
              
              <div className="backdrop-blur-xl bg-gradient-to-br from-orange-50 to-orange-100 border border-orange-200 rounded-xl p-4">
                <div className="text-xs text-orange-600 uppercase tracking-wider font-medium">Tools Used</div>
                <div className="text-3xl font-bold text-orange-900 mt-1">{metrics.toolCalls.length}</div>
              </div>
            </div>

            {/* Markdown Content */}
            <div className="bg-gray-50 rounded-lg p-4 border border-gray-200">
              <div className="space-y-2">
                <div className="flex items-center gap-2 mb-3">
                  <span className="text-sm font-medium text-gray-700">Model:</span>
                  <Badge variant="secondary" className="bg-blue-100 text-blue-800">
                    {metrics.model}
                  </Badge>
                </div>
                
                {metrics.toolCalls.length > 0 && (
                  <>
                    <h3 className="text-lg font-semibold text-gray-700 mt-4 mb-2">🛠️ Tool Usage</h3>
                    <p className="text-sm text-gray-600 mb-3">
                      <strong>{successfulTools}</strong> successful, <strong>{failedTools}</strong> failed
                    </p>
                    
                    <div className="space-y-2 max-h-64 overflow-y-auto">
                      {metrics.toolCalls.map((tool, idx) => (
                        <div
                          key={idx}
                          className="flex items-center justify-between p-3 bg-white border border-gray-200 rounded-lg shadow-sm"
                        >
                          <div className="flex-1 min-w-0">
                            <div className="font-medium text-gray-800 text-sm truncate">{tool.name}</div>
                            <div className="text-xs text-gray-500">
                              {new Date(tool.timestamp).toLocaleTimeString()}
                            </div>
                          </div>
                          <div className="flex items-center gap-2 ml-3">
                            <span className="text-xs text-gray-600">{tool.duration?.toFixed(2)}s</span>
                            <Badge 
                              variant={
                                tool.status === 'completed' ? 'default' : 
                                tool.status === 'error' ? 'destructive' : 
                                'secondary'
                              } 
                              className="text-xs"
                            >
                              {tool.status}
                            </Badge>
                          </div>
                        </div>
                      ))}
                    </div>
                  </>
                )}
              </div>
            </div>
          </div>
        </div>
      </DialogContent>
    </Dialog>
  );
}

