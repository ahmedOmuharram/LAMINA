import { useState, useRef, type FormEvent } from 'react';
import { Button } from '@/components/ui/button';
import { Textarea } from '@/components/ui/textarea';

interface ChatInputProps {
  onSend: (message: string, images?: File[]) => void;
  disabled?: boolean;
  onStop?: () => void;
  isGenerating?: boolean;
}

export function ChatInput({ onSend, disabled, onStop, isGenerating }: ChatInputProps) {
  const [message, setMessage] = useState('');
  const [images, setImages] = useState<File[]>([]);
  const fileInputRef = useRef<HTMLInputElement>(null);

  const handleSubmit = (e: FormEvent) => {
    e.preventDefault();
    if (!message.trim() && images.length === 0) return;
    if (disabled) return;

    onSend(message, images);
    setMessage('');
    setImages([]);
  };

  const handleKeyDown = (e: React.KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSubmit(e);
    }
  };

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const files = Array.from(e.target.files || []);
    setImages(prev => [...prev, ...files]);
  };

  const removeImage = (index: number) => {
    setImages(prev => prev.filter((_, i) => i !== index));
  };

  return (
    <div className="backdrop-blur-2xl bg-white/30 border border-gray-200/30 rounded-3xl p-4 shadow-xl">
      {images.length > 0 && (
        <div className="flex gap-2 mb-3 overflow-x-auto pb-2">
          {images.map((file, idx) => (
            <div key={idx} className="relative shrink-0">
              <img
                src={URL.createObjectURL(file)}
                alt={`Upload ${idx + 1}`}
                className="h-20 w-20 rounded-2xl object-cover border border-gray-200/50 shadow-md"
              />
              <Button
                size="sm"
                variant="destructive"
                className="absolute -top-2 -right-2 h-6 w-6 rounded-full p-0 shadow-lg"
                onClick={() => removeImage(idx)}
              >
                ×
              </Button>
            </div>
          ))}
        </div>
      )}

      <form onSubmit={handleSubmit} className="flex items-center gap-2">
        <input
          ref={fileInputRef}
          type="file"
          accept="image/*"
          multiple
          className="hidden"
          onChange={handleFileChange}
        />

        <Button
          type="button"
          variant="ghost"
          size="icon"
          onClick={() => fileInputRef.current?.click()}
          disabled={disabled}
          className="shrink-0 hover:bg-gray-100/50 h-11 w-11 self-end"
          title="Attach images"
        >
          <svg
            xmlns="http://www.w3.org/2000/svg"
            width="20"
            height="20"
            viewBox="0 0 24 24"
            fill="none"
            stroke="currentColor"
            strokeWidth="2"
            strokeLinecap="round"
            strokeLinejoin="round"
            className="text-gray-600"
          >
            <rect width="18" height="18" x="3" y="3" rx="2" ry="2" />
            <circle cx="9" cy="9" r="2" />
            <path d="m21 15-3.086-3.086a2 2 0 0 0-2.828 0L6 21" />
          </svg>
        </Button>

        <Textarea
          value={message}
          onChange={(e) => setMessage(e.target.value)}
          onKeyDown={handleKeyDown}
          placeholder="Type your message..."
          className="flex-1 min-h-[44px] max-h-[200px] bg-white/40 border-gray-200/30 text-gray-800 placeholder:text-gray-400 resize-none shadow-inner backdrop-blur-sm py-3"
          disabled={disabled}
          rows={1}
        />

        {isGenerating ? (
          <Button 
            type="button" 
            variant="destructive" 
            onClick={onStop}
            className="shrink-0 shadow-lg h-11 self-end"
          >
            <svg className="w-4 h-4 mr-2" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 10a1 1 0 011-1h4a1 1 0 011 1v4a1 1 0 01-1 1h-4a1 1 0 01-1-1v-4z" />
            </svg>
            Stop
          </Button>
        ) : (
          <Button 
            type="submit" 
            disabled={disabled || (!message.trim() && images.length === 0)}
            className="shrink-0 bg-gradient-to-r from-[#0b63c1] to-[#47b9ff] hover:from-[#0a5aa8] hover:to-[#3da8e6] shadow-lg h-11 self-end"
          >
            <svg className="w-4 h-4 mr-2" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 19l9 2-9-18-9 18 9-2zm0 0v-8" />
            </svg>
            Send
          </Button>
        )}
      </form>
    </div>
  );
}
