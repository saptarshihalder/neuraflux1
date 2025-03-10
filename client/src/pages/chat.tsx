import { useEffect, useState, useRef } from "react";
import { Card } from "@/components/ui/card";
import MessageList from "@/components/chat/message-list";
import MessageInput from "@/components/chat/message-input";
import { useToast } from "@/hooks/use-toast";
import { type Message } from "@shared/schema";

export default function Chat() {
  const [messages, setMessages] = useState<Message[]>([]);
  const [socket, setSocket] = useState<WebSocket | null>(null);
  const [isConnected, setIsConnected] = useState(false);
  const [isModelReady, setIsModelReady] = useState(false);
  const [isTyping, setIsTyping] = useState(false);
  const { toast } = useToast();
  const messagesEndRef = useRef<HTMLDivElement>(null);

  // Auto-scroll to bottom when messages change
  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages]);

  useEffect(() => {
    // Clear any existing messages
    setMessages([]);
    
    // Add welcome message with a delay to ensure it renders properly
    setTimeout(() => {
      setMessages([
        {
          id: Date.now(),
          role: "assistant",
          content: "Hi! I'm NeuraFlux, an AI assistant. How can I help you today? I can answer questions about math, my capabilities, or general knowledge.",
          timestamp: new Date(),
          modelOutput: {}
        }
      ]);
    }, 100);

    const protocol = window.location.protocol === "https:" ? "wss:" : "ws:";
    const wsUrl = `${protocol}//${window.location.host}/chatws`;
    const ws = new WebSocket(wsUrl);

    ws.onopen = () => {
      console.log("WebSocket connection established");
      setIsConnected(true);
      toast({
        title: "Connected",
        description: "Chat connection established"
      });
    };

    ws.onmessage = (event) => {
      const response = JSON.parse(event.data);
      console.log("Received WebSocket message:", response);
      
      if (response.type === "status") {
        setIsModelReady(response.content.ready);
        return;
      }
      
      if (response.type === "typing") {
        setIsTyping(response.content.typing);
        return;
      }
      
      if (response.type === "error") {
        toast({
          variant: "destructive",
          title: "Error",
          description: response.content
        });
        setIsTyping(false);
      } else if (response.type === "message" && response.content) {
        setIsTyping(false);
        setMessages(prev => [...prev, {
          id: Date.now(),
          role: "assistant",
          content: response.content,
          timestamp: new Date(),
          modelOutput: {}
        }]);
      }
    };

    ws.onerror = () => {
      setIsConnected(false);
      toast({
        variant: "destructive",
        title: "Connection Error",
        description: "Failed to connect to the chat server"
      });
    };

    ws.onclose = () => {
      console.log("WebSocket connection closed");
      setIsConnected(false);
      toast({
        variant: "default",
        title: "Connection Lost",
        description: "Trying to reconnect..."
      });
    };

    setSocket(ws);
    return () => ws.close();
  }, []);

  const sendMessage = (content: string) => {
    if (!content || content.trim() === '') return;
    
    if (socket?.readyState === WebSocket.OPEN) {
      const message: Message = {
        id: Date.now(),
        role: "user",
        content,
        timestamp: new Date(),
        modelOutput: {}
      };

      setMessages(prev => [...prev, message]);
      setIsTyping(true);
      
      socket.send(JSON.stringify({
        role: "user",
        content,
        modelOutput: {}
      }));
    } else {
      toast({
        variant: "destructive",
        title: "Connection Error",
        description: "Not connected to the chat server. Please try again."
      });
    }
  };

  return (
    <div className="flex flex-col h-screen bg-slate-50 dark:bg-slate-900">
      {/* Header */}
      <header className="sticky top-0 z-10 backdrop-blur-sm bg-white/75 dark:bg-slate-900/75 border-b border-slate-200 dark:border-slate-800">
        <div className="container mx-auto px-4 py-3 flex items-center justify-between">
          <h1 className="text-2xl font-bold bg-gradient-to-r from-blue-600 to-violet-600 bg-clip-text text-transparent">
            NeuraFlux Chat
          </h1>
          <div className="flex items-center space-x-2">
            <div className={`w-2 h-2 rounded-full ${isConnected ? 'bg-green-500' : 'bg-red-500'}`}></div>
            <span className="text-sm text-slate-500 dark:text-slate-400">
              {isModelReady ? 'Model Ready' : 'Model Loading...'}
            </span>
          </div>
        </div>
      </header>

      {/* Main chat area */}
      <div className="container mx-auto flex-1 overflow-hidden flex flex-col p-4">
        <Card className="flex-1 flex flex-col bg-white dark:bg-slate-800 shadow-sm rounded-xl overflow-hidden">
          <div className="flex-1 overflow-y-auto p-4">
            <MessageList messages={messages} />
            {isTyping && (
              <div className="flex items-center space-x-2 text-slate-500 dark:text-slate-400 p-4 rounded-lg bg-slate-100 dark:bg-slate-700 animate-pulse">
                <div className="w-2 h-2 rounded-full bg-current animate-bounce"></div>
                <div className="w-2 h-2 rounded-full bg-current animate-bounce delay-75"></div>
                <div className="w-2 h-2 rounded-full bg-current animate-bounce delay-150"></div>
              </div>
            )}
            <div ref={messagesEndRef} />
          </div>
          
          <div className="border-t border-slate-200 dark:border-slate-700 p-4 backdrop-blur-sm bg-white/50 dark:bg-slate-800/50">
            <MessageInput 
              onSend={sendMessage} 
              disabled={!isConnected || !isModelReady}
              placeholder={!isConnected ? "Connecting..." : !isModelReady ? "Model is loading..." : "Type your message..."}
            />
            <p className="text-xs text-center mt-2 text-slate-500 dark:text-slate-400">
              NeuraFlux can answer questions about mathematics, general knowledge, and information about itself.
            </p>
          </div>
        </Card>
      </div>
    </div>
  );
}