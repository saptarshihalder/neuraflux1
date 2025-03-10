import { type Message } from "@shared/schema";
import { ScrollArea } from "@/components/ui/scroll-area";
import { cn } from "@/lib/utils";
import { User, Bot } from "lucide-react";

interface MessageListProps {
  messages: Message[];
}

export default function MessageList({ messages }: MessageListProps) {
  // Function to format code blocks in messages
  const formatMessage = (content: string) => {
    if (!content || content.trim() === '') {
      return <span className="italic text-gray-500">Empty message</span>;
    }
    
    // Split the message by code blocks
    const parts = content.split(/(```[\s\S]*?```)/g);
    
    return parts.map((part, index) => {
      // Check if this part is a code block
      if (part.startsWith('```') && part.endsWith('```')) {
        // Extract language and code
        const match = part.match(/```(\w*)\n([\s\S]*?)```/);
        const language = match?.[1] || '';
        const code = match?.[2] || part.slice(3, -3);
        
        return (
          <pre key={index} className="bg-slate-800 text-slate-100 p-3 rounded-md my-2 overflow-x-auto">
            {language && (
              <div className="text-xs text-slate-400 mb-1">{language}</div>
            )}
            <code>{code}</code>
          </pre>
        );
      }
      
      // Handle normal text with line breaks
      return (
        <span key={index} className="whitespace-pre-wrap">
          {part}
        </span>
      );
    });
  };

  return (
    <ScrollArea className="flex-1">
      <div className="flex flex-col gap-6 p-4">
        {messages.map((message) => (
          <div
            key={message.id}
            className={cn(
              "flex gap-3",
              message.role === "user" ? "justify-end" : "justify-start"
            )}
          >
            {message.role !== "user" && (
              <div className="w-8 h-8 rounded-full bg-blue-100 flex items-center justify-center text-blue-600">
                <Bot size={18} />
              </div>
            )}
            
            <div
              className={cn(
                "rounded-lg p-4 max-w-[80%] shadow-sm",
                message.role === "user"
                  ? "bg-blue-600 text-white"
                  : "bg-white text-gray-800 dark:bg-slate-800 dark:text-white border border-gray-200 dark:border-slate-700"
              )}
            >
              <div className="font-semibold text-sm mb-1">
                {message.role === "user" ? "You" : "NeuraFlux"}
              </div>
              <div className={cn(
                "text-sm",
                message.role === "user" 
                  ? "text-white" 
                  : "text-gray-800 dark:text-gray-200"
              )}>
                {formatMessage(message.content)}
              </div>
            </div>
            
            {message.role === "user" && (
              <div className="w-8 h-8 rounded-full bg-blue-600 flex items-center justify-center text-white">
                <User size={18} />
              </div>
            )}
          </div>
        ))}
      </div>
    </ScrollArea>
  );
}
