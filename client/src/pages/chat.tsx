import { useEffect, useState } from "react";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { MessageList } from "@/components/chat/message-list";
import { MessageInput } from "@/components/chat/message-input";
import { AlertCircle, Info } from "lucide-react";
import { Alert, AlertDescription, AlertTitle } from "@/components/ui/alert";
import { useToast } from "@/hooks/use-toast";
import { type Message } from "@shared/schema";

export default function Chat() {
  const [messages, setMessages] = useState<Message[]>([]);
  const [socket, setSocket] = useState<WebSocket | null>(null);
  const [connected, setConnected] = useState(false);
  const [currentType, setCurrentType] = useState<string>("general");
  const { toast } = useToast();

  useEffect(() => {
    const protocol = window.location.protocol === "https:" ? "wss:" : "ws:";
    const wsUrl = `${protocol}//${window.location.host}/chatws`;
    const ws = new WebSocket(wsUrl);

    ws.onopen = () => {
      console.log("Connected to server");
      setConnected(true);
      setSocket(ws);
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
          modelOutput: {},
          type: response.type
        }]);
      }
    };

    ws.onerror = (error) => {
      console.error("WebSocket error:", error);
      toast({
        variant: "destructive",
        title: "Connection Error",
        description: "Failed to connect to the chat server"
      });
    };

    ws.onclose = () => {
      console.log("Disconnected from server");
      setConnected(false);
      setSocket(null);
      toast({
        variant: "default",
        title: "Connection Lost",
        description: "Trying to reconnect..."
      });
    };

    return () => {
      ws.close();
    };
  }, []);

  const sendMessage = (content: string, type: string) => {
    if (!socket || socket.readyState !== WebSocket.OPEN) {
      console.error("WebSocket is not connected");
      return;
    }

    setCurrentType(type);
    
    // Format message based on type
    let formattedMessage = content;
    if (type === "math") {
      formattedMessage = `[MATH QUESTION] ${content}`;
    } else if (type === "grammar") {
      formattedMessage = `[GRAMMAR QUESTION] ${content}`;
    }
    
    socket.send(formattedMessage);
    setMessages(prev => [...prev, { role: "user", content, type }]);
  };

  // Helper function to get tips based on current type
  const getTypeTips = () => {
    switch (currentType) {
      case "math":
        return (
          <Alert variant="default" className="mb-4">
            <AlertCircle className="h-4 w-4" />
            <AlertTitle>Math Mode</AlertTitle>
            <AlertDescription>
              You can ask questions like "What is 125 + 37?" or "Calculate 15% of 230".
            </AlertDescription>
          </Alert>
        );
      case "grammar":
        return (
          <Alert variant="default" className="mb-4">
            <AlertCircle className="h-4 w-4" />
            <AlertTitle>Grammar Mode</AlertTitle>
            <AlertDescription>
              You can ask for grammar rules, check sentences, or ask for corrections like "Is this sentence correct: She don't like apples".
            </AlertDescription>
          </Alert>
        );
      default:
        return null;
    }
  };

  return (
    <div className="container mx-auto max-w-4xl h-screen p-4 flex flex-col">
      <Card className="flex-1 p-6 flex flex-col gap-4 overflow-hidden bg-gradient-to-b from-background to-muted">
        <CardHeader>
          <CardTitle className="text-3xl font-bold mb-4 bg-gradient-to-r from-primary to-primary/70 bg-clip-text text-transparent">
            NeuraFlex Chat
          </CardTitle>
          <CardDescription>
            A hybrid transformer language model with grammar and math capabilities.
          </CardDescription>
        </CardHeader>
        <CardContent>
          {getTypeTips()}
          <MessageList messages={messages} />
          <div className="mt-4">
            <MessageInput onSend={sendMessage} />
          </div>
        </CardContent>
      </Card>
    </div>
  );
}