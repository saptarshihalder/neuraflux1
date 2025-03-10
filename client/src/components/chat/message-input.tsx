import { useState, KeyboardEvent } from "react";
import { Button } from "@/components/ui/button";
import { Textarea } from "@/components/ui/textarea";
import { Send } from "lucide-react";

interface MessageInputProps {
  onSend: (message: string) => void;
  disabled?: boolean;
  placeholder?: string;
}

export default function MessageInput({ onSend, disabled = false, placeholder = "Type your message..." }: MessageInputProps) {
  const [message, setMessage] = useState("");

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    if (message.trim() && !disabled) {
      onSend(message);
      setMessage("");
    }
  };

  const handleKeyDown = (e: KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSubmit(e);
    }
  };

  // Helper text to show examples of supported capabilities
  const getHelperText = () => {
    return (
      <div className="text-xs text-gray-500 mt-1">
        Try asking about: sequences (3, 6, 9, _, 15), integrals (log 1+x), or simple math (5 + 7)
      </div>
    );
  };

  return (
    <div>
      <form onSubmit={handleSubmit} className="flex gap-2">
        <Textarea
          value={message}
          onChange={(e) => setMessage(e.target.value)}
          placeholder={placeholder}
          className="resize-none focus:border-blue-500 focus:ring-1 focus:ring-blue-500"
          rows={3}
          disabled={disabled}
          onKeyDown={handleKeyDown}
        />
        <Button 
          type="submit" 
          size="icon"
          disabled={disabled || !message.trim()}
          className="self-end bg-blue-600 hover:bg-blue-700 transition-colors"
        >
          <Send className="h-4 w-4" />
        </Button>
      </form>
      {!disabled && getHelperText()}
    </div>
  );
}
