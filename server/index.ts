import express, { type Request, Response, NextFunction } from "express";
import cors from 'cors';
import { createServer } from 'http';
import { registerRoutes } from './routes';
import { setupVite, serveStatic, log } from "./vite";

const PORT = process.env.PORT || 3001;

async function main() {
  try {
    const app = express();
    
    // Middleware
    app.use(cors());
    app.use(express.json());
    app.use(express.urlencoded({ extended: false }));
    
    // Setup logging middleware
    app.use((req, res, next) => {
      const start = Date.now();
      const path = req.path;
      let capturedJsonResponse: Record<string, any> | undefined = undefined;
    
      const originalResJson = res.json;
      res.json = function (bodyJson, ...args) {
        capturedJsonResponse = bodyJson;
        return originalResJson.apply(res, [bodyJson, ...args]);
      };
    
      res.on("finish", () => {
        const duration = Date.now() - start;
        if (path.startsWith("/api")) {
          let logLine = `${req.method} ${path} ${res.statusCode} in ${duration}ms`;
          if (capturedJsonResponse) {
            logLine += ` :: ${JSON.stringify(capturedJsonResponse)}`;
          }
    
          if (logLine.length > 80) {
            logLine = logLine.slice(0, 79) + "…";
          }
    
          log(logLine);
        }
      });
    
      next();
    });
    
    // Create HTTP server
    const server = createServer(app);
    
    // Register routes and WebSocket handlers
    registerRoutes(app, server);
    
    // Error handling middleware
    app.use((err: any, _req: Request, res: Response, _next: NextFunction) => {
      const status = err.status || err.statusCode || 500;
      const message = err.message || "Internal Server Error";
    
      res.status(status).json({ message });
      throw err;
    });
    
    // Setup Vite or serve static files based on environment
    if (app.get("env") === "development") {
      await setupVite(app, server);
    } else {
      serveStatic(app);
    }
    
    // Start the server
    const port = parseInt(PORT as string, 10);
    server.listen({
      port,
      host: "0.0.0.0",
      reusePort: true,
    }, () => {
      log(`Server running on http://localhost:${port}`);
    });
  } catch (error) {
    console.error('Failed to start server:', error);
    process.exit(1);
  }
}

main();
