// DOM Elements
const chatBox = document.getElementById('chat-box');
const userInput = document.getElementById('user-input');
const sendButton = document.getElementById('send-button');
const typingIndicator = document.getElementById('typing-indicator');
const minimizeBtn = document.getElementById('minimize-btn');
const chatContainer = document.getElementById('chat-container');
const themeToggle = document.getElementById('theme-toggle');
const particlesCanvas = document.getElementById('particles-canvas');
const ctx = particlesCanvas.getContext('2d');

let messageId = 1;
let particles = [];
let mouse = { x: null, y: null };

// Initialize canvas size
function resizeCanvas() {
    particlesCanvas.width = window.innerWidth;
    particlesCanvas.height = window.innerHeight;
}

resizeCanvas();
window.addEventListener('resize', resizeCanvas);

// Particle class for interactive background
class Particle {
    constructor() {
        this.x = Math.random() * particlesCanvas.width;
        this.y = Math.random() * particlesCanvas.height;
        this.size = Math.random() * 3 + 1;
        this.speedX = Math.random() * 2 - 1;
        this.speedY = Math.random() * 2 - 1;
        this.opacity = Math.random() * 0.5 + 0.2;
    }

    update() {
        this.x += this.speedX;
        this.y += this.speedY;

        // Bounce off edges
        if (this.x > particlesCanvas.width || this.x < 0) {
            this.speedX *= -1;
        }
        if (this.y > particlesCanvas.height || this.y < 0) {
            this.speedY *= -1;
        }

        // Mouse interaction
        if (mouse.x !== null && mouse.y !== null) {
            const dx = mouse.x - this.x;
            const dy = mouse.y - this.y;
            const distance = Math.sqrt(dx * dx + dy * dy);
            
            if (distance < 150) {
                const force = (150 - distance) / 150;
                const angle = Math.atan2(dy, dx);
                this.x -= Math.cos(angle) * force * 3;
                this.y -= Math.sin(angle) * force * 3;
            }
        }
    }

    draw() {
        ctx.fillStyle = `rgba(255, 255, 255, ${this.opacity})`;
        ctx.beginPath();
        ctx.arc(this.x, this.y, this.size, 0, Math.PI * 2);
        ctx.fill();
    }
}

// Initialize particles
function initParticles() {
    particles = [];
    const numberOfParticles = 100;
    for (let i = 0; i < numberOfParticles; i++) {
        particles.push(new Particle());
    }
}

// Animate particles
function animateParticles() {
    ctx.clearRect(0, 0, particlesCanvas.width, particlesCanvas.height);
    
    particles.forEach(particle => {
        particle.update();
        particle.draw();
    });

    // Connect nearby particles
    for (let i = 0; i < particles.length; i++) {
        for (let j = i + 1; j < particles.length; j++) {
            const dx = particles[i].x - particles[j].x;
            const dy = particles[i].y - particles[j].y;
            const distance = Math.sqrt(dx * dx + dy * dy);

            if (distance < 100) {
                const opacity = (1 - distance / 100) * 0.2;
                ctx.strokeStyle = `rgba(255, 255, 255, ${opacity})`;
                ctx.lineWidth = 1;
                ctx.beginPath();
                ctx.moveTo(particles[i].x, particles[i].y);
                ctx.lineTo(particles[j].x, particles[j].y);
                ctx.stroke();
            }
        }
    }

    requestAnimationFrame(animateParticles);
}

// Track mouse position
window.addEventListener('mousemove', (e) => {
    mouse.x = e.x;
    mouse.y = e.y;
});

window.addEventListener('mouseout', () => {
    mouse.x = null;
    mouse.y = null;
});

// Initialize particles animation
initParticles();
animateParticles();

// Theme toggle functionality
function toggleTheme() {
    const currentTheme = document.documentElement.getAttribute('data-theme');
    const newTheme = currentTheme === 'dark' ? 'light' : 'dark';
    
    document.documentElement.setAttribute('data-theme', newTheme);
    localStorage.setItem('theme', newTheme);
    
    // Update icon
    const icon = themeToggle.querySelector('i');
    if (newTheme === 'dark') {
        icon.classList.remove('fa-moon');
        icon.classList.add('fa-sun');
    } else {
        icon.classList.remove('fa-sun');
        icon.classList.add('fa-moon');
    }
}

// Load saved theme
function loadTheme() {
    const savedTheme = localStorage.getItem('theme') || 'light';
    document.documentElement.setAttribute('data-theme', savedTheme);
    
    const icon = themeToggle.querySelector('i');
    if (savedTheme === 'dark') {
        icon.classList.remove('fa-moon');
        icon.classList.add('fa-sun');
    }
}

// Initialize on page load
loadTheme();
loadWelcomeTopic();

themeToggle.addEventListener('click', toggleTheme);

// Load dynamic topic for welcome message
async function loadWelcomeTopic() {
    try {
        const response = await fetch('/topic');
        if (response.ok) {
            const data = await response.json();
            const topic = data.topic || "the ingested content";
            const welcomeMessage = document.getElementById('welcome-message');
            if (welcomeMessage) {
                welcomeMessage.textContent = `Hello! I'm your Docker RAG chatbot. Ask me anything about ${topic}.`;
            }
        }
    } catch (error) {
        console.error('Error loading topic:', error);
        // Fallback to generic message
        const welcomeMessage = document.getElementById('welcome-message');
        if (welcomeMessage) {
            welcomeMessage.textContent = "Hello! I'm your Docker RAG chatbot. Ask me anything!";
        }
    }
}

// Format time
function getTimeString() {
    const now = new Date();
    const hours = now.getHours().toString().padStart(2, '0');
    const minutes = now.getMinutes().toString().padStart(2, '0');
    return `${hours}:${minutes}`;
}

// Create message element
function createMessage(content, isUser = false) {
    const messageDiv = document.createElement('div');
    messageDiv.classList.add('message');
    messageDiv.classList.add(isUser ? 'user-message' : 'bot-message');
    messageDiv.setAttribute('data-message-id', messageId++);

    const avatarClass = isUser ? 'user-avatar' : 'bot-avatar';
    const iconClass = isUser ? 'fa-user' : 'fa-robot';
    
    messageDiv.innerHTML = `
        <div class="avatar ${avatarClass}">
            <i class="fa-solid ${iconClass}"></i>
        </div>
        <div class="bubble-wrapper">
            <div class="bubble">
                <p>${content}</p>
            </div>
            <div class="message-time">${getTimeString()}</div>
        </div>
    `;

    return messageDiv;
}

// Show typing indicator
function showTypingIndicator() {
    typingIndicator.classList.add('active');
    chatBox.scrollTop = chatBox.scrollHeight;
}

// Hide typing indicator
function hideTypingIndicator() {
    typingIndicator.classList.remove('active');
}

// Display message
function displayMessage(content, isUser = false) {
    const messageElement = createMessage(content, isUser);
    chatBox.appendChild(messageElement);
    
    // Smooth scroll
    setTimeout(() => {
        chatBox.scrollTo({
            top: chatBox.scrollHeight,
            behavior: 'smooth'
        });
    }, 100);

    return messageElement;
}

// Send message
async function sendMessage() {
    const message = userInput.value.trim();
    
    if (message === '') {
        // Shake animation if empty
        userInput.style.animation = 'shake 0.5s';
        setTimeout(() => {
            userInput.style.animation = '';
        }, 500);
        return;
    }

    // Display user message
    displayMessage(message, true);
    
    // Clear input
    userInput.value = '';
    userInput.style.height = 'auto';

    // Disable send button
    sendButton.disabled = true;
    sendButton.style.opacity = '0.6';

    // Show typing indicator
    showTypingIndicator();

    try {
        // Simulate minimal network delay
        await new Promise(resolve => setTimeout(resolve, 300));

        // Send request to backend
        const response = await fetch('/query', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({ query: message }),
        });

        if (!response.ok) {
            throw new Error(`HTTP error! Status: ${response.status}`);
        }

        const data = await response.json();

        // Hide typing indicator
        hideTypingIndicator();

        // Display bot response
        displayMessage(data.answer, false);

    } catch (error) {
        console.error('Error:', error);
        hideTypingIndicator();
        
        // Display styled error message
        const errorMsg = `Sorry, an error occurred. Please try again. 😔`;
        displayMessage(errorMsg, false);
    } finally {
        // Re-enable send button
        sendButton.disabled = false;
        sendButton.style.opacity = '1';
        userInput.focus();
    }
}

// Send button click event
sendButton.addEventListener('click', sendMessage);

// Enter key event
userInput.addEventListener('keypress', (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
        e.preventDefault();
        sendMessage();
    }
});

// Auto-expand input field
userInput.addEventListener('input', function() {
    this.style.height = 'auto';
    const maxHeight = 120;
    if (this.scrollHeight <= maxHeight) {
        this.style.height = this.scrollHeight + 'px';
    } else {
        this.style.height = maxHeight + 'px';
    }
});

// Minimize/maximize chat
minimizeBtn.addEventListener('click', () => {
    chatContainer.classList.toggle('minimized');
    const icon = minimizeBtn.querySelector('i');
    
    if (chatContainer.classList.contains('minimized')) {
        icon.classList.remove('fa-chevron-down');
        icon.classList.add('fa-chevron-up');
        chatContainer.style.height = '80px';
    } else {
        icon.classList.remove('fa-chevron-up');
        icon.classList.add('fa-chevron-down');
        chatContainer.style.height = '85vh';
    }
});

// Add shake animation
const style = document.createElement('style');
style.textContent = `
    @keyframes shake {
        0%, 100% { transform: translateX(0); }
        10%, 30%, 50%, 70%, 90% { transform: translateX(-5px); }
        20%, 40%, 60%, 80% { transform: translateX(5px); }
    }
`;
document.head.appendChild(style);

