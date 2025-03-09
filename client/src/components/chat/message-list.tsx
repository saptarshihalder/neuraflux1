import { ScrollArea } from "../ui/scroll-area";
import { cn } from "@/lib/utils";

interface Message {
  role: 'user' | 'assistant';
  content: string;
  type?: string;
}

export interface MessageListProps {
  messages: Message[];
}

export function MessageList({ messages }: MessageListProps) {
  return (
    <ScrollArea className="flex-1 pr-4">
      <div className="flex flex-col gap-4">
        {messages.length === 0 ? (
          <div className="flex-1 flex items-center justify-center text-center">
            <div className="text-muted-foreground">
              Send a message to start chatting
            </div>
          </div>
        ) : (
          messages.map((message, index) => (
            <div
              key={index}
              className={cn(
                "p-4 rounded-lg",
                message.role === "user"
                  ? "bg-primary/10 ml-auto"
                  : "bg-muted"
              )}
            >
              <div className="font-semibold mb-1">
                {message.role === "user" ? "You" : "NeuraFlux"}
                {message.type && message.role === "user" && (
                  <span className="ml-2 text-xs font-normal text-muted-foreground">
                    {message.type.charAt(0).toUpperCase() + message.type.slice(1)} question
                  </span>
                )}
              </div>
              <div className="whitespace-pre-wrap">{message.content}</div>
            </div>
          ))
        )}
      </div>
    </ScrollArea>
  );
}
