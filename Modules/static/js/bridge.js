
// Define addChatMessage first since it's used by event handlers
function addChatMessage(sender, message, type) {
    const chatMessages = document.getElementById('chat-messages');
    if (!chatMessages) return;
    const messageDiv = document.createElement('div');
    messageDiv.className = `chat-message ${type}`;
    messageDiv.innerHTML = `<strong>${sender}:</strong> ${message}`;
    chatMessages.appendChild(messageDiv);
    chatMessages.scrollTop = chatMessages.scrollHeight;
}

// WebSocket connection
const wsUrl = `ws://${window.location.host}/ws`;
console.log('Connecting to WebSocket:', wsUrl);
const socket = new WebSocket(wsUrl);

socket.onopen = () => {
    console.log('✅ WebSocket connected');
    addChatMessage('System', 'Connected to AI Core', 'system');
};

socket.onerror = (error) => {
    console.error('❌ WebSocket error:', error);
    addChatMessage('System', 'Connection error', 'error');
};

socket.onclose = () => {
    console.log('🔌 WebSocket disconnected');
    addChatMessage('System', 'Disconnected from AI Core', 'error');
};

socket.onmessage = (event) => {
    const data = JSON.parse(event.data);
    console.log('📨 Received:', data.type);
    
    if (data.type === 'chat_response') {
        addChatMessage('AI', data.data.response, 'ai');
    } else if (data.type === 'chat_error') {
        addChatMessage('System', `Error: ${data.data.error}`, 'error');
    } else if (data.type === 'ai_started') {
        addChatMessage('System', 'AI engine started successfully!', 'system');
        addChatMessage('AI', data.data.test_response, 'ai');
    } else if (data.type === 'ai_error') {
        addChatMessage('System', `AI startup error: ${data.data.message}`, 'error');
    } else if (data.type === 'metrics_update') {
        updateMetrics(data);
    } else {
        updateModuleStatus(data);
    }
};

function updateMetrics(data) {
    document.querySelector('#cpu-usage').textContent = `CPU: ${data.metrics.cpu_usage}%`;
    document.querySelector('#memory-usage').textContent = `Memory: ${data.metrics.memory_usage}%`;
    document.querySelector('#ai-status').textContent = `Status: ${data.ai_status.status}`;
    document.querySelector('#uptime').textContent = `Uptime: ${data.ai_status.uptime.formatted}`;
    document.querySelector('#active-targets').textContent = `Active Targets: ${data.attack_metrics.active_targets}`;
    document.querySelector('#success-rate').textContent = `Success Rate: ${(data.attack_metrics.success_rate * 100).toFixed(2)}%`;

    const modulesList = document.querySelector('#modules-list');
    if (modulesList && data.active_modules) {
        modulesList.innerHTML = data.active_modules.map(module => `
            <div class="module-item" onclick="loadModule('${module.name}')" data-name="${module.name}">
                ${module.name} - ${module.status}
                ${module.status === 'running' ? 
                    `(PID: ${module.pid}, CPU: ${module.cpu}%, Memory: ${module.memory.toFixed(2)} MB)` : 
                    '<button class="load-btn">Load</button>'
                }
            </div>
        `).join('');
    }
}

function updateModuleStatus(data) {
    if (data.type === "module_loaded" && data.data && data.data.status === "success") {
        const moduleElement = document.querySelector(`[data-name="${data.data.module_name}"]`);
        if (moduleElement) {
            const btn = moduleElement.querySelector('.load-btn');
            if (btn) {
                btn.textContent = 'Running';
                btn.classList.add('running');
            }
        }
    }
}

function loadModule(moduleName) {
    socket.send(JSON.stringify({ command: 'load_module', module_name: moduleName }));
}

function startAI() {
    addChatMessage('System', 'Starting AI engine...', 'system');
    socket.send(JSON.stringify({ command: 'start_ai' }));
}

function sendChatMessage() {
    const input = document.getElementById('chat-input');
    const message = input.value.trim();
    if (message && socket.readyState === WebSocket.OPEN) {
        addChatMessage('You', message, 'user');
        socket.send(JSON.stringify({ command: 'chat', message: message }));
        input.value = '';
    }
}

function handleChatKeyPress(event) {
    if (event.key === 'Enter') {
        sendChatMessage();
    }
}
