import WebSocket from 'ws';

// Connect to the WebSocket server
const ws = new WebSocket('ws://localhost:3001/chat');

// Array of test grammar queries
const testQueries = [
  // Grammar questions
  '[GRAMMAR QUESTION] What is subject-verb agreement?',
  '[GRAMMAR QUESTION] Can you explain the subjunctive mood?',
  '[GRAMMAR QUESTION] What is a dangling modifier?',
  
  // Grammar corrections
  '[GRAMMAR QUESTION] Is this sentence correct: She don\'t like apples',
  '[GRAMMAR QUESTION] Check this: me and my friend went to the store',
  '[GRAMMAR QUESTION] Correct this: I should of done my homework',
  
  // Math questions for comparison
  '[MATH QUESTION] What is 125 + 37?',
  '[MATH QUESTION] Calculate 15% of 230',
  
  // General knowledge
  'What is your name?',
  'What can you help me with?'
];

// Current query index
let currentIndex = 0;

// When the connection is open
ws.on('open', function open() {
  console.log('Connected to WebSocket server');
  
  // Send the first query
  if (currentIndex < testQueries.length) {
    console.log(`Sending: ${testQueries[currentIndex]}`);
    ws.send(testQueries[currentIndex]);
  }
});

// Handle messages
ws.on('message', function incoming(data) {
  const response = data.toString();
  console.log(`\nResponse for "${testQueries[currentIndex]}":`);
  console.log(response);
  console.log('-'.repeat(80));
  
  // Move to the next query
  currentIndex++;
  
  // If there are more queries, send the next one after a delay
  if (currentIndex < testQueries.length) {
    setTimeout(() => {
      console.log(`Sending: ${testQueries[currentIndex]}`);
      ws.send(testQueries[currentIndex]);
    }, 1000); // 1 second delay
  } else {
    console.log('All test queries completed');
    ws.close();
  }
});

// Handle errors
ws.on('error', function error(err) {
  console.error('WebSocket error:', err);
});

// Handle connection close
ws.on('close', function close() {
  console.log('Connection closed');
  process.exit(0);
}); 