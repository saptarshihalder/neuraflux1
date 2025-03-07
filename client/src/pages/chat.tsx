import { useEffect, useState } from "react";
import { Card } from "@/components/ui/card";
import MessageList from "@/components/chat/message-list";
import MessageInput from "@/components/chat/message-input";
import { useToast } from "@/hooks/use-toast";
import { type Message } from "@shared/schema";

export default function Chat() {
  const [messages, setMessages] = useState<Message[]>([]);
  const [socket, setSocket] = useState<WebSocket | null>(null);
  const { toast } = useToast();

  useEffect(() => {
    const protocol = window.location.protocol === "https:" ? "wss:" : "ws:";
    const wsUrl = `${protocol}//${window.location.host}/chatws`;
    const ws = new WebSocket(wsUrl);

    ws.onopen = () => {
      console.log("WebSocket connection established");
    };

    ws.onmessage = (event) => {
      const response = JSON.parse(event.data);
      if (response.type === "error") {
        toast({
          variant: "destructive",
          title: "Error",
          description: response.content
        });
      } else {
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
      toast({
        variant: "destructive",
        title: "Connection Error",
        description: "Failed to connect to the chat server"
      });
    };

    ws.onclose = () => {
      console.log("WebSocket connection closed");
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
    if (socket?.readyState === WebSocket.OPEN) {
      const message: Message = {
        id: Date.now(),
        role: "user",
        content,
        timestamp: new Date(),
        modelOutput: {}
      };

      setMessages(prev => [...prev, message]);
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
    <div className="container mx-auto max-w-4xl h-screen p-4 flex flex-col">
      <Card className="flex-1 p-6 flex flex-col gap-4 overflow-hidden bg-gradient-to-b from-background to-muted">
        <h1 className="text-3xl font-bold mb-4 bg-gradient-to-r from-primary to-primary/70 bg-clip-text text-transparent">
          NeuraFlex Chat
        </h1>
        <p className="text-sm text-muted-foreground mb-4">
          Ask me anything! I can help with math calculations and general questions.
        </p>
        <MessageList messages={messages} />
        <MessageInput onSend={sendMessage} />
      </Card>
    </div>
  );
}