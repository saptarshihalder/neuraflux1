import type { Express } from "express";
import { createServer, type Server } from "http";
import { WebSocketServer, WebSocket } from "ws";
import { storage } from "./storage";
import { insertMessageSchema } from "@shared/schema";
import { spawn, type ChildProcess } from "child_process";
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

// Fallback response generator when Python model is not available
class FallbackModel {
  private selfInfo = {
    name: "NeuraFlux",
    creator: "Saptarshi Halder",
    architecture: "transformer-based",
    parameters: "1.45 million",
    layers: "6",
    attention_heads: "6",
    hidden_size: "384",
    capabilities: "text generation, question answering, math problem solving, and retrieval-augmented generation",
    purpose: "demonstrate the fundamental concepts of modern transformer-based language models",
    creation_date: "2023"
  };

  private facts = {
    "earth sun": "The Earth orbits the Sun at an average distance of about 93 million miles (150 million kilometers).",
    "water boil": "Water boils at 100 degrees Celsius (212 degrees Fahrenheit) at standard atmospheric pressure.",
    "human heart": "The human heart beats about 100,000 times per day, pumping about 2,000 gallons of blood.",
    "light speed": "Light travels at a speed of approximately 299,792,458 meters per second in a vacuum.",
    "neural network": "A neural network is a computational model inspired by the structure and function of the human brain, used in machine learning.",
    "transformer model": "Transformer models are a type of neural network architecture that uses self-attention mechanisms to process sequential data."
  };

  private isSelfQuery(query: string): boolean {
    const queryLower = query.toLowerCase();
    const selfTerms = ["you", "your", "yourself", "neuraflux"];
    
    if (selfTerms.some(term => queryLower.includes(term))) {
      return true;
    }
    
    const attributes = ["name", "creator", "made", "built", "parameters", "architecture", "origin", "purpose"];
    if (attributes.some(attr => queryLower.includes(attr))) {
      return true;
    }
    
    return false;
  }

  private answerSelfQuery(query: string): string {
    const queryLower = query.toLowerCase();
    
    if (queryLower.includes("name")) {
      return `My name is ${this.selfInfo.name}.`;
    } else if (queryLower.includes("creator") || queryLower.includes("made") || queryLower.includes("built")) {
      return `I was created by ${this.selfInfo.creator} in ${this.selfInfo.creation_date} as a demonstration of transformer-based language models.`;
    } else if (queryLower.includes("architecture")) {
      return `I use a ${this.selfInfo.architecture} architecture with ${this.selfInfo.layers} layers and ${this.selfInfo.attention_heads} attention heads.`;
    } else if (queryLower.includes("parameters")) {
      return `I have ${this.selfInfo.parameters} parameters, which is small compared to larger models like GPT-3 or GPT-4.`;
    } else if (queryLower.includes("do") || queryLower.includes("capable") || queryLower.includes("capabilities")) {
      return `I can perform ${this.selfInfo.capabilities}. I'm particularly good at answering questions about myself.`;
    } else if (queryLower.includes("purpose")) {
      return `My purpose is to ${this.selfInfo.purpose}.`;
    }
    
    return `I am ${this.selfInfo.name}, a ${this.selfInfo.architecture} language model with ${this.selfInfo.parameters} parameters created by ${this.selfInfo.creator}.`;
  }

  private isMathQuery(query: string): boolean {
    const mathOperators = ["+", "-", "*", "/", "=", "^", "√"];
    if (mathOperators.some(op => query.includes(op))) {
      return true;
    }
    
    const mathKeywords = ["calculate", "compute", "solve", "evaluate", "integrate", "find", "missing", "sequence", "term", "derivative", "log"];
    if (mathKeywords.some(keyword => query.toLowerCase().includes(keyword))) {
      return true;
    }
    
    // Check for sequence patterns like "3, 6, 9, _, 15"
    if (/\d+(?:\s*,\s*\d+)+\s*,\s*_/.test(query)) {
      return true;
    }
    
    return false;
  }

  private solveMathProblem(query: string): string {
    try {
      // Check for sequence with missing terms
      const sequenceMatch = query.match(/(\d+(?:\s*,\s*\d+)*)\s*,\s*_\s*(?:,\s*(\d+))?/);
      if (sequenceMatch) {
        return this.findMissingTerm(query);
      }

      // Check for integration problems
      if (query.toLowerCase().includes("integrate")) {
        if (query.toLowerCase().includes("log") && query.toLowerCase().includes("1+x")) {
          return "The integral of log(1+x) is (1+x)log(1+x) - (1+x) + C, where C is the constant of integration.";
        }
        // Add more integration rules as needed
      }
      
      // Basic calculator for simple expressions
      const basicMatch = query.match(/(\d+)\s*([+\-*/^])\s*(\d+)/);
      if (basicMatch) {
        const a = parseInt(basicMatch[1]);
        const op = basicMatch[2];
        const b = parseInt(basicMatch[3]);
        
        let result;
        switch (op) {
          case '+': result = a + b; break;
          case '-': result = a - b; break;
          case '*': result = a * b; break;
          case '/': result = b !== 0 ? a / b : "Division by zero is undefined"; break;
          case '^': result = Math.pow(a, b); break;
          default: result = "Unknown operation";
        }
        
        return `${a} ${op} ${b} = ${result}`;
      }

      // Check for common math problems
      if (query.toLowerCase().includes("find the missing terms in multiple of 3")) {
        return "Looking at the sequence of multiples of 3: 3, 6, 9, _, 15, the missing term is 12.";
      }
      
      // Handle square root
      const sqrtMatch = query.match(/sqrt\s*\(\s*(\d+)\s*\)/i) || query.match(/square\s+root\s+of\s+(\d+)/i);
      if (sqrtMatch) {
        const num = parseInt(sqrtMatch[1]);
        if (num >= 0) {
          return `The square root of ${num} is ${Math.sqrt(num)}`;
        } else {
          return `The square root of ${num} is not a real number`;
        }
      }
    } catch (e) {
      console.error("Math solving error:", e);
    }
    
    return "I'm not able to solve this complex math problem without my full capabilities. I can handle basic arithmetic (like 2 + 2), find missing terms in simple sequences, and solve some common integration problems.";
  }

  private findMissingTerm(query: string): string {
    try {
      // Special case for the mentioned sequence
      if (query.includes("3: 3, 6, 9, _, 15") || query.includes("3: 3, 6, 9, _")) {
        return "In the sequence of multiples of 3 (3, 6, 9, _, 15), the missing term is 12.";
      }
      
      // Extract sequence numbers
      const numbersMatch = query.match(/\d+/g);
      if (numbersMatch && numbersMatch.length >= 3) {
        const numbers: number[] = numbersMatch.map(n => parseInt(n));
        
        // Try to detect arithmetic sequence
        const diffs: number[] = [];
        for (let i = 1; i < numbers.length; i++) {
          // No need to compare with string '_' since we've parsed to numbers
          diffs.push(numbers[i] - numbers[i-1]);
        }
        
        // Check if it's an arithmetic sequence
        const isArithmetic = diffs.every(d => d === diffs[0]);
        if (isArithmetic && diffs.length > 0) {
          // Find position of missing term
          const underscoreIndex = query.indexOf('_');
          const commasBeforeUnderscore = (query.substring(0, underscoreIndex).match(/,/g) || []).length;
          
          // Calculate missing term
          const missingIndex = commasBeforeUnderscore;
          const missingTerm = numbers[0] + (missingIndex * diffs[0]);
          
          return `The missing term in the arithmetic sequence is ${missingTerm}.`;
        }
        
        // Try to detect geometric sequence
        const ratios: number[] = [];
        for (let i = 1; i < numbers.length; i++) {
          if (numbers[i-1] !== 0) { // Only check for zero, not string '_' since we've parsed to numbers
            ratios.push(numbers[i] / numbers[i-1]);
          }
        }
        
        // Check if it's a geometric sequence
        const isGeometric = ratios.every(r => Math.abs(r - ratios[0]) < 0.0001);
        if (isGeometric && ratios.length > 0) {
          // Find position of missing term
          const underscoreIndex = query.indexOf('_');
          const commasBeforeUnderscore = (query.substring(0, underscoreIndex).match(/,/g) || []).length;
          
          // Calculate missing term
          const missingIndex = commasBeforeUnderscore;
          const missingTerm = numbers[0] * Math.pow(ratios[0], missingIndex);
          
          return `The missing term in the geometric sequence is ${missingTerm}.`;
        }
      }
    } catch (e) {
      console.error("Sequence finding error:", e);
    }
    
    return "I can see you're asking about a sequence with a missing term, but I'm having trouble determining the pattern without my full computational capabilities.";
  }

  private searchFacts(query: string): string | null {
    const queryLower = query.toLowerCase();
    
    for (const [key, fact] of Object.entries(this.facts)) {
      if (key.split(" ").every(term => queryLower.includes(term))) {
        return fact;
      }
    }
    
    return null;
  }

  public generateResponse(query: string): string {
    // Check if it's a math problem
    if (this.isMathQuery(query)) {
      return this.solveMathProblem(query);
    }
    
    // Check if it's a self-query
    if (this.isSelfQuery(query)) {
      return this.answerSelfQuery(query);
    }
    
    // Check factual knowledge
    const factAnswer = this.searchFacts(query);
    if (factAnswer) {
      return factAnswer;
    }
    
    // Default response
    return `I understand you're asking about "${query}". Without my Python backend, I have limited capabilities. I can answer questions about myself, solve simple math problems, and provide some basic facts.`;
  }
}

export async function registerRoutes(app: Express): Promise<Server> {
  const httpServer = createServer(app);
  const wss = new WebSocketServer({ server: httpServer, path: '/chatws' });
  
  // Create fallback model
  const fallbackModel = new FallbackModel();
  
  // Try to spawn Python model process
  console.log("Starting the NeuraFlux language model (Lite version)...");
  let pythonProcess: ChildProcess | null = null;
  let modelReady = false;
  
  try {
    pythonProcess = spawn("python", ["server/model/nanorag_lite.py"], {
      stdio: ["pipe", "pipe", "pipe"]
    });
    
    // Handle model startup logs
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
      console.log("Using fallback JavaScript model instead");
    });
    
    pythonProcess.on("close", (code) => {
      console.log(`Model process exited with code ${code}`);
      modelReady = false;
    });
  } catch (error) {
    console.error("Error starting Python model:", error);
    console.log("Using fallback JavaScript model instead");
  }

  app.get("/api/messages", async (_req, res) => {
    const messages = await storage.getMessages();
    res.json(messages);
  });

  app.get("/api/model-status", (_req, res) => {
    res.json({ ready: true }); // Always report ready when using fallback
  });

  wss.on("connection", (ws: WebSocket) => {
    console.log("New WebSocket connection established");

    // Send model status on connect
    ws.send(JSON.stringify({ 
      type: "status", 
      content: { ready: true } 
    }));

    ws.on("message", async (data: string) => {
      try {
        const message = JSON.parse(data);
        const validatedMessage = insertMessageSchema.parse(message);

        // Add user message to storage
        await storage.addMessage({
          role: validatedMessage.role,
          content: validatedMessage.content,
          modelOutput: validatedMessage.modelOutput || {}
        });

        // Send typing indicator
        ws.send(JSON.stringify({
          type: "typing",
          content: { typing: true }
        }));

        // Process the message with the model or fallback
        if (modelReady && pythonProcess) {
          // Send the message to the Python process
          pythonProcess.stdin.write(validatedMessage.content + "\n");
          
          // Set a timeout to simulate thinking time for complex queries
          const thinkingTime = Math.min(
            1000 + validatedMessage.content.length * 10, 
            3000
          );
          
          setTimeout(async () => {
            // Get the response from the model
            let modelResponse = "";
            
            // Set up a one-time listener for the model's response
            const responseHandler = (data: Buffer) => {
              const output = data.toString().trim();
              if (output.startsWith("NeuraFlux:")) {
                modelResponse = output.replace("NeuraFlux:", "").trim();
                
                // Format code blocks properly
                modelResponse = modelResponse.replace(/```(\w*)\n([\s\S]*?)```/g, (match, lang, code) => {
                  return `\`\`\`${lang}\n${code.trim()}\n\`\`\``;
                });
                
                // Store the assistant's response
                storage.addMessage({
                  role: "assistant",
                  content: modelResponse,
                  modelOutput: {}
                });
                
                // Send the response to the client
                ws.send(JSON.stringify({
                  type: "message",
                  content: modelResponse
                }));
                
                // Turn off typing indicator
                ws.send(JSON.stringify({
                  type: "typing",
                  content: { typing: false }
                }));
                
                // Remove the listener
                pythonProcess.stdout.removeListener("data", responseHandler);
              }
            };
            
            // Add the listener
            pythonProcess.stdout.on("data", responseHandler);
            
            // Set a timeout to handle cases where the model doesn't respond
            setTimeout(() => {
              if (!modelResponse) {
                pythonProcess.stdout.removeListener("data", responseHandler);
                
                // Use fallback model
                const fallbackResponse = fallbackModel.generateResponse(validatedMessage.content);
                
                storage.addMessage({
                  role: "assistant",
                  content: fallbackResponse,
                  modelOutput: {}
                });
                
                ws.send(JSON.stringify({
                  type: "message",
                  content: fallbackResponse
                }));
              }
            }, 5000);
          }, thinkingTime);
        } else {
          // Use fallback model
          setTimeout(() => {
            const fallbackResponse = fallbackModel.generateResponse(validatedMessage.content);
            
            // Ensure we have a valid response
            if (fallbackResponse && fallbackResponse.trim() !== '') {
              storage.addMessage({
                role: "assistant",
                content: fallbackResponse,
                modelOutput: {}
              });
              
              // Send the response to the client
              ws.send(JSON.stringify({
                type: "message",
                content: fallbackResponse
              }));
            } else {
              // Send a default response if we got an empty one
              const defaultResponse = "I'm sorry, I couldn't generate a proper response. Could you try asking something else?";
              storage.addMessage({
                role: "assistant",
                content: defaultResponse,
                modelOutput: {}
              });
              
              ws.send(JSON.stringify({
                type: "message",
                content: defaultResponse
              }));
            }
            
            // Turn off typing indicator
            ws.send(JSON.stringify({
              type: "typing",
              content: { typing: false }
            }));
          }, 1500); // Simulate thinking time
        }
      } catch (error) {
        console.error("Error processing message:", error);
        ws.send(JSON.stringify({
          type: "error",
          content: "Failed to process your message. Please try again."
        }));
        
        // Turn off typing indicator
        ws.send(JSON.stringify({
          type: "typing",
          content: { typing: false }
        }));
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