import { useEffect, useState } from 'react';
import {
  Dialog,
  DialogContent,
  DialogHeader,
  DialogTitle,
} from '@/components/ui/dialog';
import { Button } from '@/components/ui/button';
import { Checkbox } from '@/components/ui/checkbox';
import { ScrollArea } from '@/components/ui/scroll-area';
import { apiClient } from '@/lib/api';
import type { AIFunctionInfo } from '@/types/api';

interface SettingsModalProps {
  open: boolean;
  onOpenChange: (open: boolean) => void;
  onSettingsChange?: (enabledFunctions: string[]) => void;
}

const STORAGE_KEY = 'lamina_enabled_functions';

// Load enabled functions from localStorage
function loadEnabledFunctions(allFunctions: AIFunctionInfo[]): Set<string> {
  try {
    const stored = localStorage.getItem(STORAGE_KEY);
    if (stored) {
      const parsed = JSON.parse(stored);
      return new Set(parsed);
    }
  } catch (e) {
    console.error('Failed to load enabled functions from localStorage:', e);
  }
  // Default: all functions enabled
  return new Set(allFunctions.map(f => f.name));
}

// Save enabled functions to localStorage
function saveEnabledFunctions(enabledFunctions: Set<string>) {
  try {
    localStorage.setItem(STORAGE_KEY, JSON.stringify(Array.from(enabledFunctions)));
  } catch (e) {
    console.error('Failed to save enabled functions to localStorage:', e);
  }
}

export function SettingsModal({ open, onOpenChange, onSettingsChange }: SettingsModalProps) {
  const [functions, setFunctions] = useState<AIFunctionInfo[]>([]);
  const [enabledFunctions, setEnabledFunctions] = useState<Set<string>>(new Set());
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  // Fetch functions on mount
  useEffect(() => {
    async function fetchFunctions() {
      try {
        setLoading(true);
        setError(null);
        const response = await apiClient.getAIFunctions();
        setFunctions(response.functions);
        
        // Load enabled functions from localStorage
        const enabled = loadEnabledFunctions(response.functions);
        setEnabledFunctions(enabled);
        
        // Notify parent of initial settings
        if (onSettingsChange) {
          onSettingsChange(Array.from(enabled));
        }
      } catch (err) {
        console.error('Failed to fetch AI functions:', err);
        setError('Failed to load AI functions. Please try again.');
      } finally {
        setLoading(false);
      }
    }

    fetchFunctions();
  }, []);

  // Notify parent when settings change
  useEffect(() => {
    if (onSettingsChange) {
      onSettingsChange(Array.from(enabledFunctions));
    }
  }, [enabledFunctions, onSettingsChange]);

  // Group functions by category
  const functionsByCategory = functions.reduce((acc, func) => {
    if (!acc[func.category]) {
      acc[func.category] = [];
    }
    acc[func.category].push(func);
    return acc;
  }, {} as Record<string, AIFunctionInfo[]>);

  const categories = Object.keys(functionsByCategory).sort();

  const handleToggleFunction = (funcName: string) => {
    setEnabledFunctions(prev => {
      const newSet = new Set(prev);
      if (newSet.has(funcName)) {
        newSet.delete(funcName);
      } else {
        newSet.add(funcName);
      }
      saveEnabledFunctions(newSet);
      return newSet;
    });
  };

  const handleToggleAll = (enable: boolean) => {
    const newSet = enable ? new Set(functions.map(f => f.name)) : new Set<string>();
    setEnabledFunctions(newSet);
    saveEnabledFunctions(newSet);
  };

  const allEnabled = enabledFunctions.size === functions.length;
  const noneEnabled = enabledFunctions.size === 0;

  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      <DialogContent className="max-w-4xl max-h-[85vh] overflow-hidden flex flex-col">
        <DialogHeader>
          <DialogTitle className="text-xl">AI Function Settings</DialogTitle>
        </DialogHeader>

        <div className="flex items-center gap-3 mb-4 pb-4 border-b border-gray-200">
          <div className="text-sm text-gray-600">
            {enabledFunctions.size} of {functions.length} functions enabled
          </div>
          <div className="flex-1" />
          <Button
            variant="outline"
            size="sm"
            onClick={() => handleToggleAll(true)}
            disabled={allEnabled || loading}
            className="text-xs"
          >
            Enable All
          </Button>
          <Button
            variant="outline"
            size="sm"
            onClick={() => handleToggleAll(false)}
            disabled={noneEnabled || loading}
            className="text-xs"
          >
            Disable All
          </Button>
        </div>

        {loading && (
          <div className="flex items-center justify-center py-8">
            <div className="text-sm text-gray-600">Loading functions...</div>
          </div>
        )}

        {error && (
          <div className="flex items-center justify-center py-8">
            <div className="text-sm text-red-600">{error}</div>
          </div>
        )}

        {!loading && !error && (
          <ScrollArea className="h-[calc(85vh-240px)] pr-4">
            <div className="space-y-6 pb-4">
              {categories.map(category => (
                <div key={category}>
                  <h3 className="text-sm font-semibold text-gray-700 mb-3 flex items-center gap-2">
                    {category}
                    <span className="text-xs font-normal text-gray-500">
                      ({functionsByCategory[category].filter(f => enabledFunctions.has(f.name)).length}/
                      {functionsByCategory[category].length})
                    </span>
                  </h3>
                  <div className="space-y-3">
                    {functionsByCategory[category].map(func => (
                      <div
                        key={func.name}
                        className="flex items-start gap-3 p-3 rounded-lg hover:bg-gray-50 transition-colors"
                      >
                        <Checkbox
                          id={func.name}
                          checked={enabledFunctions.has(func.name)}
                          onCheckedChange={() => handleToggleFunction(func.name)}
                          className="mt-0.5"
                        />
                        <label
                          htmlFor={func.name}
                          className="flex-1 cursor-pointer"
                        >
                          <div className="font-medium text-sm text-gray-900">
                            {func.name}
                          </div>
                          <div className="text-xs text-gray-600 mt-0.5 leading-relaxed">
                            {func.description}
                          </div>
                        </label>
                      </div>
                    ))}
                  </div>
                </div>
              ))}
            </div>
          </ScrollArea>
        )}

        <div className="flex justify-end pt-4 border-t border-gray-200 mt-4">
          <Button onClick={() => onOpenChange(false)} size="sm">
            Close
          </Button>
        </div>
      </DialogContent>
    </Dialog>
  );
}

