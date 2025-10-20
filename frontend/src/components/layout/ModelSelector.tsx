import { useEffect, useState } from 'react';
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from '@/components/ui/select';
import { apiClient } from '@/lib/api';
import type { Model } from '@/types/api';

interface ModelSelectorProps {
  value: string;
  onChange: (value: string) => void;
}

export function ModelSelector({ value, onChange }: ModelSelectorProps) {
  const [models, setModels] = useState<Model[]>([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    apiClient
      .getModels()
      .then((response) => {
        setModels(response.data);
        setLoading(false);
      })
      .catch((error) => {
        console.error('Failed to fetch models:', error);
        setLoading(false);
      });
  }, []);

  if (loading) {
    return (
      <div className="flex items-center gap-2 text-gray-500">
        <svg className="animate-spin h-4 w-4" fill="none" viewBox="0 0 24 24">
          <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" />
          <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z" />
        </svg>
        <span className="text-sm">Loading...</span>
      </div>
    );
  }

  // Get the selected model to display in the trigger
  const selectedModel = models.find((m) => m.id === value);
  
  // Simplify model names
  const getDisplayName = (model: Model) => {
    if (model.id === 'gpt-4o-mini') return '4o-mini';
    else if (model.id === 'gpt-4o') return '4o';
    else if (model.id === 'o1') return 'o1';
    return model.name;
  };

  return (
    <div className="flex flex-col gap-2">
      <label className="text-xs font-medium text-gray-600 uppercase tracking-wider">Model</label>
      <Select value={value} onValueChange={onChange}>
        <SelectTrigger className="w-full bg-white/30 border-gray-200/30 text-gray-800 backdrop-blur-xl hover:bg-white/40 shadow-sm text-left !h-auto !min-h-[48px]">
          <div className="flex items-center gap-2 w-full overflow-hidden py-0.25">
            <img src="/gpt.svg" alt="GPT" className="w-4 h-4 flex-shrink-0" />
            {selectedModel ? (
              <div className="flex flex-col flex-1 min-w-0">
                <span className="font-medium text-sm">{getDisplayName(selectedModel)}</span>
                <span className="text-xs text-gray-500 truncate">
                  {selectedModel.description}
                </span>
              </div>
            ) : (
              <SelectValue placeholder="Select model" />
            )}
          </div>
        </SelectTrigger>
        <SelectContent className="bg-white/95 border-gray-200/50 backdrop-blur-xl shadow-xl min-w-[280px]">
          {models.map((model) => (
            <SelectItem 
              key={model.id} 
              value={model.id}
              className="text-gray-800 hover:bg-gray-100/50 focus:bg-gray-100/70 py-3"
            >
              <div className="flex flex-col w-full">
                <span className="font-medium text-sm">{getDisplayName(model)}</span>
                <span className="text-xs text-gray-500 leading-relaxed break-words">
                  {model.description}
                </span>
              </div>
            </SelectItem>
          ))}
        </SelectContent>
      </Select>
    </div>
  );
}
