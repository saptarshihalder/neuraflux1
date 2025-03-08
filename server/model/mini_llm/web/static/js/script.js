document.addEventListener('DOMContentLoaded', () => {
    // DOM Elements
    const chatMessages = document.getElementById('chat-messages');
    const userInput = document.getElementById('user-input');
    const sendButton = document.getElementById('send-button');
    const stepsToggle = document.getElementById('steps-toggle');
    const examplesList = document.getElementById('examples-list');

    
    let isWaitingForResponse = false;

    // Load example problems
    loadExamples();

    // Event listeners
    userInput.addEventListener('keypress', (e) => {
        if (e.key === 'Enter' && !e.shiftKey) {
            e.preventDefault();
            sendMessage();
        }
    });

    sendButton.addEventListener('click', sendMessage);
    
    // Handle "who are you" button if it exists
    const whoAreYouBtn = document.querySelector('.who-are-you-btn');
    if (whoAreYouBtn) {
        whoAreYouBtn.addEventListener('click', async () => {
            if (isWaitingForResponse) return;
            
            // Add user message
            addMessage("Who are you?", 'user');
            
            // Add loading indicator
            const loadingEl = addLoadingIndicator();
            
            try {
                const response = await fetch('/api/who-are-you');
                if (!response.ok) {
                    throw new Error('Failed to get response');
                }
                
                const data = await response.json();
                
                // Remove loading indicator
                loadingEl.remove();
                
                // Create a formatted response
                const formattedResponse = `
                    I am ${data.name}, a specialized mathematical language model created by ${data.creator}.
                    
                    ${data.description}
                    
                    **Architecture**: ${data.architecture}
                    **Parameters**: ${data.parameter_count}
                    **Version History**: 
                    ${data.version_history ? data.version_history.map(v => `• ${v}`).join('\n') : ''}
                    
                    **Unique Features**:
                    ${data.unique_features ? data.unique_features.map(f => `• ${f}`).join('\n') : ''}
                    
                    **Capabilities**:
                    ${data.capabilities.map(c => `• ${c}`).join('\n')}
                    
                    **Training**: ${data.training}
                    
                    **Limitations**: ${data.limitations}
                `;
                
                // Add bot response
                addBotResponse(formattedResponse, []);
                
            } catch (error) {
                console.error('Error getting info:', error);
                loadingEl.remove();
                addErrorMessage();
            }
        });
    }

    // Handle training info modal
    const trainingInfoBtn = document.querySelector('.training-info-btn');
    const trainingModal = document.getElementById('training-modal');
    const closeModal = document.querySelector('.close-modal');
    
    if (trainingInfoBtn && trainingModal) {
        trainingInfoBtn.addEventListener('click', () => {
            trainingModal.style.display = 'block';
        });
        
        // Close modal when X is clicked
        if (closeModal) {
            closeModal.addEventListener('click', () => {
                trainingModal.style.display = 'none';
            });
        }
        
        // Close modal when clicking outside
        window.addEventListener('click', (event) => {
            if (event.target === trainingModal) {
                trainingModal.style.display = 'none';
            }
        });
        
        // Close modal with Escape key
        window.addEventListener('keydown', (event) => {
            if (event.key === 'Escape' && trainingModal.style.display === 'block') {
                trainingModal.style.display = 'none';
            }
        });
    }

    // Functions
    async function loadExamples() {
        try {
            const response = await fetch('/api/examples');
            const examples = await response.json();
            
            examplesList.innerHTML = '';
            examples.forEach(example => {
                const exampleEl = document.createElement('div');
                exampleEl.className = 'example-item';
                exampleEl.textContent = example;
                exampleEl.addEventListener('click', () => {
                    userInput.value = example;
                    userInput.focus();
                });
                examplesList.appendChild(exampleEl);
            });
        } catch (error) {
            console.error('Error loading examples:', error);
        }
    }

    async function sendMessage() {
        const message = userInput.value.trim();
        if (!message || isWaitingForResponse) return;

        // Add user message to chat
        addMessage(message, 'user');
        
        // Clear input
        userInput.value = '';
        
        // Add loading indicator
        const loadingEl = addLoadingIndicator();
        
        // Disable input while waiting
        isWaitingForResponse = true;
        sendButton.disabled = true;
        
        try {
            const response = await fetch('/api/chat', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({
                    message: message,
                    show_steps: stepsToggle.checked
                })
            });
            
            if (!response.ok) {
                throw new Error('Failed to get response');
            }
            
            const data = await response.json();
            
            // Remove loading indicator
            loadingEl.remove();
            
            // Add bot response
            addBotResponse(data.solution, data.steps);
            
        } catch (error) {
            console.error('Error sending message:', error);
            loadingEl.remove();
            addErrorMessage();
        } finally {
            // Re-enable input
            isWaitingForResponse = false;
            sendButton.disabled = false;
            userInput.focus();
        }
    }

    function addMessage(message, sender) {
        const messageEl = document.createElement('div');
        messageEl.className = `message ${sender}`;
        
        const contentEl = document.createElement('div');
        contentEl.className = 'message-content';
        
        const textEl = document.createElement('p');
        textEl.textContent = message;
        
        contentEl.appendChild(textEl);
        messageEl.appendChild(contentEl);
        
        chatMessages.appendChild(messageEl);
        scrollToBottom();
    }

    function addBotResponse(solution, steps = []) {
        const messageEl = document.createElement('div');
        messageEl.className = 'message bot';
        
        const contentEl = document.createElement('div');
        contentEl.className = 'message-content';
        
        // Solution
        const solutionEl = document.createElement('div');
        solutionEl.className = 'solution';
        solutionEl.innerHTML = formatMathText(solution);
        contentEl.appendChild(solutionEl);
        
        // Steps (if available and toggle is checked)
        if (steps && steps.length > 0 && stepsToggle.checked) {
            const stepsEl = document.createElement('div');
            stepsEl.className = 'step-by-step';
            
            const stepsTitle = document.createElement('h3');
            stepsTitle.textContent = 'Step-by-step solution:';
            stepsEl.appendChild(stepsTitle);
            
            const stepsList = document.createElement('div');
            stepsList.className = 'step-list';
            
            steps.forEach(step => {
                const stepEl = document.createElement('div');
                stepEl.className = 'step';
                stepEl.innerHTML = formatMathText(step);
                stepsList.appendChild(stepEl);
            });
            
            stepsEl.appendChild(stepsList);
            contentEl.appendChild(stepsEl);
        }
        
        messageEl.appendChild(contentEl);
        chatMessages.appendChild(messageEl);
        
        // Render math expressions
        renderMathInElement(messageEl);
        
        scrollToBottom();
    }

    function addLoadingIndicator() {
        const loadingEl = document.createElement('div');
        loadingEl.className = 'message bot loading';
        
        const loadingText = document.createElement('span');
        loadingText.textContent = 'Thinking';
        
        const dots = document.createElement('div');
        dots.className = 'dots';
        
        for (let i = 0; i < 3; i++) {
            const dot = document.createElement('div');
            dot.className = 'dot';
            dots.appendChild(dot);
        }
        
        loadingEl.appendChild(loadingText);
        loadingEl.appendChild(dots);
        
        chatMessages.appendChild(loadingEl);
        scrollToBottom();
        
        return loadingEl;
    }

    function addErrorMessage() {
        const messageEl = document.createElement('div');
        messageEl.className = 'message bot error';
        
        const contentEl = document.createElement('div');
        contentEl.className = 'message-content';
        
        const textEl = document.createElement('p');
        textEl.textContent = 'Sorry, I encountered an error processing your request. Please try again.';
        
        contentEl.appendChild(textEl);
        messageEl.appendChild(contentEl);
        
        chatMessages.appendChild(messageEl);
        scrollToBottom();
    }

    function formatMathText(text) {
        // Identify inline math expressions (surrounded by $ symbols)
        // and block math expressions (surrounded by $$ symbols)
        
        // Check if the text already contains LaTeX delimiters
        if (text.includes('$') || text.includes('\\(') || text.includes('\\[')) {
            return text
                .replace(/\$\$(.*?)\$\$/g, '<div class="math-expression">\\[$1\\]</div>')
                .replace(/\$(.*?)\$/g, '<span class="math-expression">\\($1\\)</span>');
        }
        
        // For text without LaTeX delimiters, we can add them to common math expressions
        return text
            // Format exponents like x², x³, etc.
            .replace(/(\w+)\^(\d+)/g, '<span class="math-expression">\\($1^{$2}\\)</span>')
            // Format fractions like a/b
            .replace(/(\d+)\/(\d+)/g, '<span class="math-expression">\\(\\frac{$1}{$2}\\)</span>')
            // Format derivative notation
            .replace(/f'?\(x\)/g, '<span class="math-expression">\\(f\'(x)\\)</span>')
            // Format integrals
            .replace(/∫(.*?)dx/g, '<span class="math-expression">\\(\\int $1 dx\\)</span>')
            // Format square roots
            .replace(/√\((.*?)\)/g, '<span class="math-expression">\\(\\sqrt{$1}\\)</span>')
            .replace(/√([\w\d]+)/g, '<span class="math-expression">\\(\\sqrt{$1}\\)</span>');
    }

    function renderMathInElement(element) {
        // Use KaTeX to render math expressions
        try {
            if (window.renderMathInElement) {
                window.renderMathInElement(element, {
                    delimiters: [
                        {left: '\\[', right: '\\]', display: true},
                        {left: '\\(', right: '\\)', display: false}
                    ],
                    throwOnError: false
                });
            } else {
                console.warn('KaTeX auto-render not available');
            }
        } catch (error) {
            console.error('Error rendering math:', error);
        }
    }

    function scrollToBottom() {
        chatMessages.scrollTop = chatMessages.scrollHeight;
    }
}); 