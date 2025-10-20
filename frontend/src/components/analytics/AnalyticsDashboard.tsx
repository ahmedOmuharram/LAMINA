import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import type { ConversationMetrics } from '@/types/api';

interface AnalyticsDashboardProps {
  metrics: ConversationMetrics;
}

export function AnalyticsDashboard({ metrics }: AnalyticsDashboardProps) {
  const avgDuration = metrics.messageCount > 0 
    ? (metrics.totalDuration / (metrics.messageCount / 2)).toFixed(2)
    : '0';

  const successfulTools = metrics.toolCalls.filter(t => t.status === 'completed').length;
  const failedTools = metrics.toolCalls.filter(t => t.status === 'error').length;

  return (
    <div className="space-y-6 p-6">
      <h2 className="text-2xl font-semibold text-gray-800">Analytics Dashboard</h2>

      <div className="grid grid-cols-2 gap-4">
        <Card className="backdrop-blur-xl bg-white/40 border-gray-200/50 shadow-lg">
          <CardHeader className="pb-3">
            <CardDescription className="text-gray-600 text-xs uppercase tracking-wider">Messages</CardDescription>
            <CardTitle className="text-4xl text-gray-800">{metrics.messageCount}</CardTitle>
          </CardHeader>
        </Card>

        <Card className="backdrop-blur-xl bg-white/40 border-gray-200/50 shadow-lg">
          <CardHeader className="pb-3">
            <CardDescription className="text-gray-600 text-xs uppercase tracking-wider">Tokens Used</CardDescription>
            <CardTitle className="text-4xl text-gray-800">{metrics.totalTokens}</CardTitle>
          </CardHeader>
        </Card>

        <Card className="backdrop-blur-xl bg-white/40 border-gray-200/50 shadow-lg">
          <CardHeader className="pb-3">
            <CardDescription className="text-gray-600 text-xs uppercase tracking-wider">Avg Response Time</CardDescription>
            <CardTitle className="text-4xl text-gray-800">{avgDuration}s</CardTitle>
          </CardHeader>
        </Card>

        <Card className="backdrop-blur-xl bg-white/40 border-gray-200/50 shadow-lg">
          <CardHeader className="pb-3">
            <CardDescription className="text-gray-600 text-xs uppercase tracking-wider">Tool Calls</CardDescription>
            <CardTitle className="text-4xl text-gray-800">{metrics.toolCalls.length}</CardTitle>
          </CardHeader>
        </Card>
      </div>

      <Card className="backdrop-blur-xl bg-white/40 border-gray-200/50 shadow-lg">
        <CardHeader>
          <CardTitle className="text-sm text-gray-800">Current Model</CardTitle>
        </CardHeader>
        <CardContent>
          <Badge variant="secondary" className="text-base bg-gradient-to-r from-[#47b9ff]/20 to-[#0b63c1]/20 text-gray-800 border-gray-200/50">
            {metrics.model}
          </Badge>
        </CardContent>
      </Card>

      {metrics.toolCalls.length > 0 && (
        <Card className="backdrop-blur-xl bg-white/40 border-gray-200/50 shadow-lg">
          <CardHeader>
            <CardTitle className="text-sm text-gray-800">Tool Usage</CardTitle>
            <CardDescription className="text-gray-600">
              {successfulTools} successful, {failedTools} failed
            </CardDescription>
          </CardHeader>
          <CardContent>
            <div className="space-y-2 max-h-[400px] overflow-y-auto">
              {metrics.toolCalls.map((tool, idx) => (
                <div
                  key={idx}
                  className="flex items-center justify-between p-3 backdrop-blur-xl bg-white/40 border border-gray-200/50 rounded-2xl shadow-sm"
                >
                  <div className="flex-1">
                    <div className="font-medium text-gray-800 text-sm">{tool.name}</div>
                    <div className="text-xs text-gray-500">
                      {new Date(tool.timestamp).toLocaleTimeString()}
                    </div>
                  </div>
                  <div className="flex items-center gap-2">
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
          </CardContent>
        </Card>
      )}

      <Card className="backdrop-blur-xl bg-white/40 border-gray-200/50 shadow-lg">
        <CardHeader>
          <CardTitle className="text-sm text-gray-800">Total Conversation Time</CardTitle>
        </CardHeader>
        <CardContent>
          <div className="text-3xl font-bold text-gray-800">
            {(metrics.totalDuration / 1000).toFixed(2)}s
          </div>
        </CardContent>
      </Card>
    </div>
  );
}
