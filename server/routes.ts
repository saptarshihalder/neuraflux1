import type { Express } from "express";
import { createServer, type Server } from "http";
import { WebSocketServer, WebSocket } from "ws";
import { storage } from "./storage";
import { insertMessageSchema } from "@shared/schema";
import { spawn } from "child_process";
import fetch from "node-fetch";

// Remove the Perchance fallback
// const PERCHANCE_BASE_URL = "https://perchance.org/api/generators";

// async function getPerchanceResponse(prompt: string): Promise<string> {
//   try {
//     // Use more sophisticated generators based on the type of question
//     const generators = [
//       'writing-prompt-generator',
//       'ai-conversation',
//       'random-sentence'
//     ];

//     const responses = await Promise.all(
//       generators.map(async (generator) => {
//         const response = await fetch(`${PERCHANCE_BASE_URL}/${generator}/generate`);
//         if (!response.ok) return null;
//         const data = await response.json() as { text?: string };
//         return data.text;
//       })
//     );

//     // Filter out null responses and get the first valid one
//     const validResponses = responses.filter(r => r !== null);
//     if (validResponses.length > 0) {
//       return validResponses[0] || "I apologize, I'm still learning how to respond to that type of question.";
//     }

//     return "I apologize, I'm having trouble understanding that. Could you try rephrasing your question?";
//   } catch (error) {
//     console.error("Perchance API error:", error);
//     return "I apologize, I'm having trouble understanding that. Could you try rephrasing your question?";
//   }
// }

export async function registerRoutes(app: Express): Promise<Server> {
  const httpServer = createServer(app);
  const wss = new WebSocketServer({ server: httpServer, path: '/chatws' });

  // Spawn Python model process once and keep it running
  console.log("Starting the NeuraFlux language model (Lite version)...");
  const pythonProcess = spawn("python", ["server/model/nanorag_lite.py"], {
    stdio: ["pipe", "pipe", "pipe"]
  });

  // Handle model startup logs
  let modelReady = false;
  pythonProcess.stdout.on("data", (data) => {
    const output = data.toString().trim();
    console.log(`Model output: ${output}`);
    if (output.includes("NeuraFlux Lite Model")) {
      modelReady = true;
      console.log("Model initialized successfully");
    }
  });

  pythonProcess.stderr.on("data", (data) => {
    console.log(`Model log: ${data}`);
  });

  // Handle potential errors
  pythonProcess.on("error", (error) => {
    console.error("Failed to start model process:", error);
  });

  pythonProcess.on("close", (code) => {
    console.log(`Model process exited with code ${code}`);
  });

  app.get("/api/messages", async (_req, res) => {
    const messages = await storage.getMessages();
    res.json(messages);
  });

  app.get("/api/model-status", (_req, res) => {
    res.json({ ready: modelReady });
  });

  wss.on("connection", (ws: WebSocket) => {
    console.log("New WebSocket connection established");

    // Send model status on connect
    ws.send(JSON.stringify({ 
      type: "status", 
      content: { ready: modelReady } 
    }));

    ws.on("message", async (data: string) => {
      try {
        const message = JSON.parse(data);
        const validatedMessage = insertMessageSchema.parse(message);

        // Add user message to storage
        await storage.addMessage(validatedMessage);

        if (!modelReady) {
          ws.send(JSON.stringify({ 
            type: "error", 
            content: "The language model is still initializing. Please try again in a moment." 
          }));
          return;
        }

        // Send thinking indicator
        ws.send(JSON.stringify({ type: "thinking", content: true }));

        // Send to Python model for processing
        console.log("Sending query to model:", validatedMessage.content);
        pythonProcess.stdin.write(validatedMessage.content + "\n");

        // Handle model response
        pythonProcess.stdout.once("data", async (data) => {
          const response = data.toString().trim();
          console.log("Model response:", response);

          if (ws.readyState === WebSocket.OPEN) {
            // Turn off thinking indicator
            ws.send(JSON.stringify({ type: "thinking", content: false }));
            
            // Send the actual response
            ws.send(JSON.stringify({ 
              type: "response", 
              content: response 
            }));
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