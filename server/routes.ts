import Express from 'express';
import { Server as HttpServer } from 'http';
import { WebSocketServer, WebSocket } from 'ws';
import { storage } from "./storage";
import { insertMessageSchema } from "@shared/schema";
import { spawn } from 'child_process';
import path from 'path';
import fetch from 'node-fetch';

interface WebSocketWithID extends WebSocket {
    id: string;
}

interface PerchanceResponse {
    result?: string;
    [key: string]: any;
}

export const getPerchanceResponse = async (prompt: string) => {
    try {
        // Determine which generator to use based on prompt content
        let generator = 'ai-chatbot';
        
        // If using math or grammar specific tags, adjust generator
        if (prompt.toLowerCase().includes('[math question]')) {
            generator = 'calcula';
            prompt = prompt.replace(/\[math question\]/i, '').trim();
        } else if (prompt.toLowerCase().includes('[grammar question]')) {
            generator = 'grammarbot';
            prompt = prompt.replace(/\[grammar question\]/i, '').trim();
        }
        
        // Fallback to default AI chatbot for general questions
        console.log(`Using generator: ${generator} for prompt: ${prompt}`);
        
        const url = `https://perchance.org/api/1/perchance/generate/${generator}`;
        const response = await fetch(url, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                prompt
            }),
        });

        const data = await response.json() as PerchanceResponse;
        return data.result || 'I do not have a response for that.';
    } catch (error) {
        console.error('Error calling Perchance API:', error);
        return 'I encountered an error processing your request.';
    }
};

export const registerRoutes = (app: Express.Express, server: HttpServer) => {
    // Set up WebSocket server
    const wss = new WebSocketServer({ server });

    // Set up Python model process
    const modelProcess = spawn('python', [path.join(__dirname, 'model', 'nanorag.py')]);
    console.log('Started Python model process');

    modelProcess.stderr.on('data', (data) => {
        console.log(`Model stderr: ${data}`);
    });

    // Set up WebSocket connection handler
    wss.on('connection', (ws: WebSocketWithID) => {
        ws.id = Math.random().toString(36).substring(2, 10);
        console.log(`Client ${ws.id} connected`);

        // Handle incoming messages
        ws.on('message', async (message: string) => {
            try {
                const text = message.toString();

                // Check if it's a special math or grammar question format
                let processedMessage = text;
                let responsePrefix = '';
                if (text.startsWith('[MATH QUESTION]')) {
                    processedMessage = text.replace('[MATH QUESTION]', '').trim();
                    responsePrefix = '[Math] ';
                    console.log('Processing math question:', processedMessage);
                } else if (text.startsWith('[GRAMMAR QUESTION]')) {
                    processedMessage = text.replace('[GRAMMAR QUESTION]', '').trim();
                    responsePrefix = '[Grammar] ';
                    console.log('Processing grammar question:', processedMessage);
                }

                // Send the message to Python model
                modelProcess.stdin.write(processedMessage + '\n');

                // Set up a listener for the Python model's response
                const responseListener = (data: Buffer) => {
                    const response = data.toString().trim();
                    console.log(`Model response for client ${ws.id}: ${response}`);
                    if (response) {
                        ws.send(responsePrefix + response);
                        modelProcess.stdout.removeListener('data', responseListener);
                    }
                };

                modelProcess.stdout.on('data', responseListener);

                // Set a timeout to handle no response from model
                const timeoutId = setTimeout(() => {
                    modelProcess.stdout.removeListener('data', responseListener);
                    console.log(`No response from model for client ${ws.id}, using Perchance fallback`);
                    
                    // Fallback to Perchance API if no response from Python model
                    getPerchanceResponse(processedMessage).then(perchanceResponse => {
                        ws.send(responsePrefix + perchanceResponse);
                    });
                }, 5000); // 5 second timeout
                
                // Clear timeout if client disconnects
                ws.on('close', () => {
                    clearTimeout(timeoutId);
                    modelProcess.stdout.removeListener('data', responseListener);
                });
            } catch (error) {
                console.error('Error processing message:', error);
                ws.send('I encountered an error processing your request.');
            }
        });

        // Handle client disconnection
        ws.on('close', () => {
            console.log(`Client ${ws.id} disconnected`);
        });

        // Handle WebSocket errors
        ws.on('error', (error) => {
            console.error(`WebSocket error for client ${ws.id}:`, error);
        });
    });

    // Add routes
    app.get('/api/health', (req, res) => {
        res.json({ status: 'ok' });
    });

    app.get("/api/messages", async (_req, res) => {
        const messages = await storage.getMessages();
        res.json(messages);
    });
};