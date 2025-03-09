const { execSync } = require('child_process');
const path = require('path');
const fs = require('fs');

// Ensure directories exist
const grammarDir = path.join(__dirname, 'server', 'model', 'data', 'grammar');
if (!fs.existsSync(grammarDir)) {
  fs.mkdirSync(grammarDir, { recursive: true });
  console.log('Created grammar data directory');
}

// Start the server
try {
  console.log('Starting NeuraFlux server with math and grammar capabilities...');
  execSync('npm run dev', { stdio: 'inherit' });
} catch (error) {
  console.error('Failed to start server:', error);
  process.exit(1);
} 