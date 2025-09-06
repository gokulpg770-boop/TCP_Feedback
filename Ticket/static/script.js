class TicketSystemApp {
    constructor() {
        this.sessionId = this.generateSessionId();
        this.currentCategory = null;
        this.isLoading = false;
        
        this.initializeElements();
        this.attachEventListeners();
        this.loadCategories();
    }
    
    generateSessionId() {
        return 'session-' + Math.random().toString(36).substr(2, 9) + '-' + Date.now();
    }
    
    initializeElements() {
        // Screens
        this.categoryScreen = document.getElementById('category-screen');
        this.chatScreen = document.getElementById('chat-screen');
        this.successScreen = document.getElementById('success-screen');
        
        // Category elements
        this.categoryButtons = document.getElementById('category-buttons');
        
        // Chat elements
        this.selectedCategoryTitle = document.getElementById('selected-category');
        this.chatMessages = document.getElementById('chat-messages');
        this.messageInput = document.getElementById('message-input');
        this.sendBtn = document.getElementById('send-btn');
        
        // Action buttons
        this.restartBtn = document.getElementById('restart-btn');
        this.newTicketBtn = document.getElementById('new-ticket-btn');
        this.closeBtn = document.getElementById('close-btn');
        
        // Success elements
        this.successMessage = document.getElementById('success-message');
        
        // Loading spinner
        this.loadingSpinner = document.getElementById('loading-spinner');
    }
    
    attachEventListeners() {
        // Send button
        this.sendBtn.addEventListener('click', () => this.sendMessage());
        
        // Enter key in input
        this.messageInput.addEventListener('keypress', (e) => {
            if (e.key === 'Enter' && !this.isLoading) {
                this.sendMessage();
            }
        });
        
        // Restart button
        this.restartBtn.addEventListener('click', () => this.restart());
        
        // New ticket button
        this.newTicketBtn.addEventListener('click', () => this.restart());
        
        // Close button
        this.closeBtn.addEventListener('click', () => this.restart());
        
        // Input character counter
        this.messageInput.addEventListener('input', (e) => {
            const remaining = 500 - e.target.value.length;
            if (remaining < 0) {
                e.target.value = e.target.value.substring(0, 500);
            }
        });
    }
    
    async loadCategories() {
        try {
            const response = await fetch('/categories');
            const data = await response.json();
            
            this.renderCategories(data.categories);
        } catch (error) {
            console.error('Error loading categories:', error);
            this.showError('Failed to load categories. Please refresh the page.');
        }
    }
    
    renderCategories(categories) {
        const categoryIcons = {
            payroll: 'fas fa-dollar-sign',
            hr: 'fas fa-users',
            it: 'fas fa-laptop',
            facilities: 'fas fa-building',
            finance: 'fas fa-chart-line'
        };
        
        this.categoryButtons.innerHTML = categories.map(category => `
            <button class="category-btn" data-category="${category}">
                <i class="${categoryIcons[category] || 'fas fa-question'}"></i>
                ${category}
            </button>
        `).join('');
        
        // Add click listeners to category buttons
        this.categoryButtons.querySelectorAll('.category-btn').forEach(btn => {
            btn.addEventListener('click', (e) => {
                const category = e.currentTarget.dataset.category;
                this.selectCategory(category);
            });
        });
    }
    
    async selectCategory(category) {
        this.currentCategory = category;
        this.selectedCategoryTitle.textContent = `${category.charAt(0).toUpperCase() + category.slice(1)} Support`;
        
        this.showScreen('chat');
        this.showLoading(true);
        
        try {
            const response = await fetch('/chat/init', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    category: category,
                    session_id: this.sessionId
                })
            });
            
            if (!response.ok) {
                throw new Error(`HTTP error! status: ${response.status}`);
            }
            
            const data = await response.json();
            this.addMessage(data.message, 'assistant');
            
            if (data.conversation_complete) {
                this.handleConversationComplete(data.message);
            }
        } catch (error) {
            console.error('Error initializing chat:', error);
            this.showError('Failed to start conversation. Please try again.');
        } finally {
            this.showLoading(false);
        }
    }
    
    async sendMessage() {
        const message = this.messageInput.value.trim();
        if (!message || this.isLoading) return;
        
        // Add user message to chat
        this.addMessage(message, 'user');
        this.messageInput.value = '';
        
        this.showLoading(true);
        
        try {
            const response = await fetch('/chat/message', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    message: message,
                    session_id: this.sessionId
                })
            });
            
            if (!response.ok) {
                throw new Error(`HTTP error! status: ${response.status}`);
            }
            
            const data = await response.json();
            this.addMessage(data.message, 'assistant');
            
            if (data.conversation_complete) {
                this.handleConversationComplete(data.message);
            }
        } catch (error) {
            console.error('Error sending message:', error);
            this.showError('Failed to send message. Please try again.');
        } finally {
            this.showLoading(false);
        }
    }
    
    handleConversationComplete(message) {
        if (message.includes('✅ Ticket Created!')) {
            this.successMessage.textContent = message;
            this.showScreen('success');
        } else {
            // Conversation ended without ticket creation
            setTimeout(() => {
                this.restart();
            }, 2000);
        }
    }
    
    addMessage(content, sender) {
        const messageDiv = document.createElement('div');
        messageDiv.className = `message ${sender}`;
        messageDiv.textContent = content;
        
        this.chatMessages.appendChild(messageDiv);
        this.chatMessages.scrollTop = this.chatMessages.scrollHeight;
    }
    
    showScreen(screenName) {
        document.querySelectorAll('.screen').forEach(screen => {
            screen.classList.remove('active');
        });
        
        switch(screenName) {
            case 'category':
                this.categoryScreen.classList.add('active');
                break;
            case 'chat':
                this.chatScreen.classList.add('active');
                break;
            case 'success':
                this.successScreen.classList.add('active');
                break;
        }
    }
    
    showLoading(show) {
        this.isLoading = show;
        this.sendBtn.disabled = show;
        this.messageInput.disabled = show;
        
        if (show) {
            this.loadingSpinner.classList.add('active');
        } else {
            this.loadingSpinner.classList.remove('active');
        }
    }
    
    showError(message) {
        this.addMessage(`❌ Error: ${message}`, 'assistant');
    }
    
    async restart() {
        // End current session
        try {
            await fetch(`/chat/${this.sessionId}`, {
                method: 'DELETE'
            });
        } catch (error) {
            console.error('Error ending session:', error);
        }
        
        // Reset state
        this.sessionId = this.generateSessionId();
        this.currentCategory = null;
        this.chatMessages.innerHTML = '';
        this.messageInput.value = '';
        this.successMessage.textContent = '';
        
        // Show category screen
        this.showScreen('category');
        this.showLoading(false);
    }
}

// Initialize the app when DOM is loaded
document.addEventListener('DOMContentLoaded', () => {
    new TicketSystemApp();
});
