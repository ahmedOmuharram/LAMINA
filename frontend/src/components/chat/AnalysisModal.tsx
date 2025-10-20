import {
  Dialog,
  DialogContent,
  DialogHeader,
  DialogTitle,
} from '@/components/ui/dialog';

interface AnalysisModalProps {
  analysis: string;
  title?: string;
  open: boolean;
  onOpenChange: (open: boolean) => void;
}

export function AnalysisModal({ analysis, title, open, onOpenChange }: AnalysisModalProps) {
  const renderMarkdownLine = (line: string, idx: number) => {
    // Skip empty lines
    if (!line.trim()) {
      return <div key={idx} className="h-2" />;
    }

    // Headers
    if (line.startsWith('### ')) {
      return (
        <h3 key={idx} className="text-lg font-semibold text-gray-800 mt-4 mb-2">
          {line.replace('### ', '')}
        </h3>
      );
    }
    if (line.startsWith('## ')) {
      return (
        <h2 key={idx} className="text-xl font-bold text-gray-900 mt-5 mb-3">
          {line.replace('## ', '')}
        </h2>
      );
    }

    // List items
    if (line.match(/^[-*]\s/)) {
      const content = line.replace(/^[-*]\s/, '');
      // Handle bold **text**
      let formatted = content.replace(/\*\*([^*]+)\*\*/g, '<strong class="font-semibold text-gray-900">$1</strong>');
      // Handle code `text`
      formatted = formatted.replace(/`([^`]+)`/g, '<code class="px-1.5 py-0.5 bg-blue-100 text-blue-800 rounded text-sm font-mono">$1</code>');
      // Handle inline math $...$
      formatted = formatted.replace(/\$([^$]+)\$/g, '<em class="italic text-gray-700">$1</em>');
      
      return (
        <li key={idx} className="ml-6 mb-1.5 text-gray-700 leading-relaxed" dangerouslySetInnerHTML={{ __html: formatted }} />
      );
    }

    // Regular paragraphs
    let formatted = line;
    // Handle bold **text**
    formatted = formatted.replace(/\*\*([^*]+)\*\*/g, '<strong class="font-semibold text-gray-900">$1</strong>');
    // Handle code `text`
    formatted = formatted.replace(/`([^`]+)`/g, '<code class="px-1.5 py-0.5 bg-blue-100 text-blue-800 rounded text-sm font-mono">$1</code>');
    // Handle inline math $...$
    formatted = formatted.replace(/\$([^$]+)\$/g, '<em class="italic text-gray-700">$1</em>');
    
    return (
      <p key={idx} className="mb-2 text-gray-700 leading-relaxed" dangerouslySetInnerHTML={{ __html: formatted }} />
    );
  };

  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      <DialogContent className="max-w-4xl max-h-[85vh] overflow-hidden flex flex-col">
        <DialogHeader>
          <DialogTitle className="text-xl flex items-center gap-2">
            <svg className="w-5 h-5 text-blue-600" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 19v-6a2 2 0 00-2-2H5a2 2 0 00-2 2v6a2 2 0 002 2h2a2 2 0 002-2zm0 0V9a2 2 0 012-2h2a2 2 0 012 2v10m-6 0a2 2 0 002 2h2a2 2 0 002-2m0 0V5a2 2 0 012-2h2a2 2 0 012 2v14a2 2 0 01-2 2h-2a2 2 0 01-2-2z" />
            </svg>
            {title || 'Phase Diagram Analysis'}
          </DialogTitle>
        </DialogHeader>

        <div className="flex-1 overflow-y-auto pr-2">
          <div className="bg-gradient-to-br from-blue-50 to-purple-50 rounded-lg p-6 border border-blue-200">
            <div className="prose prose-sm max-w-none">
              {analysis.split('\n').map((line, idx) => renderMarkdownLine(line, idx))}
            </div>
          </div>
        </div>
      </DialogContent>
    </Dialog>
  );
}

