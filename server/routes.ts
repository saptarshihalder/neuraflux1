import type { Express } from "express";
import { createServer, type Server } from "http";
import { WebSocketServer, WebSocket } from "ws";
import { storage } from "./storage";
import { insertMessageSchema } from "@shared/schema";
import { spawn } from "child_process";
import fetch from "node-fetch";

const PERCHANCE_BASE_URL = "https://perchance.org/api/generators";

async function getPerchanceResponse(prompt: string): Promise<string> {
  try {
    // Use more sophisticated generators based on the type of question
    const generators = [
      'writing-prompt-generator',
      'ai-conversation',
      'random-sentence'
    ];

    const responses = await Promise.all(
      generators.map(async (generator) => {
        const response = await fetch(`${PERCHANCE_BASE_URL}/${generator}/generate`);
        if (!response.ok) return null;
        const data = await response.json() as { text?: string };
        return data.text;
      })
    );

    // Filter out null responses and get the first valid one
    const validResponses = responses.filter(r => r !== null);
    if (validResponses.length > 0) {
      return validResponses[0] || "I apologize, I'm still learning how to respond to that type of question.";
    }

    return "I apologize, I'm having trouble understanding that. Could you try rephrasing your question?";
  } catch (error) {
    console.error("Perchance API error:", error);
    return "I apologize, I'm having trouble understanding that. Could you try rephrasing your question?";
  }
}

export async function registerRoutes(app: Express): Promise<Server> {
  const httpServer = createServer(app);
  const wss = new WebSocketServer({ server: httpServer, path: '/chatws' });

  // Spawn Python model process once and keep it running
  const pythonProcess = spawn("python3", ["server/model/nanorag.py"], {
    stdio: ["pipe", "pipe", "pipe"]
  });

  pythonProcess.stderr.on("data", (data) => {
    console.log(`Model log: ${data}`);
  });

  app.get("/api/messages", async (_req, res) => {
    const messages = await storage.getMessages();
    res.json(messages);
  });

  wss.on("connection", (ws: WebSocket) => {
    console.log("New WebSocket connection established");

    ws.on("message", async (data: string) => {
      try {
        const message = JSON.parse(data);
        const validatedMessage = insertMessageSchema.parse(message);

        // Add user message to storage
        await storage.addMessage(validatedMessage);

        // Send to Python model for processing
        pythonProcess.stdin.write(validatedMessage.content + "\n");

        // Handle model response
        pythonProcess.stdout.once("data", async (data) => {
          if (ws.readyState === WebSocket.OPEN) {
            const response = data.toString().trim();
            console.log("Model response:", response);

            if (response.includes("I don't know") || response.includes("I'm sorry")) {
              // Fallback to Perchance for unknown questions
              const perchanceResponse = await getPerchanceResponse(validatedMessage.content);
              ws.send(JSON.stringify({ type: "response", content: perchanceResponse }));
            } else {
              ws.send(JSON.stringify({ type: "response", content: response }));
            }
          }
        });

      } catch (error: any) {
        console.error("WebSocket error:", error);
        if (ws.readyState === WebSocket.OPEN) {
          ws.send(JSON.stringify({ 
            type: "error", 
            content: "Something went wrong. Please try again." 
          }));
        }
      }
    });

    ws.on("error", (error) => {
      console.error("WebSocket error:", error);
    });

    ws.on("close", () => {
      console.log("WebSocket connection closed");
    });
  });

  return httpServer;
}