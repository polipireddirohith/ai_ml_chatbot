import React, { useState, useEffect, useRef } from 'react';
import './ChatInterface.css';

const MODELS = [
    { id: 'general', name: 'General AI Assistant' },
    { id: 'mlp', name: 'MLP (Neural Network)' },
    { id: 'cnn', name: 'CNN (Computer Vision)' },
    { id: 'rnn', name: 'RNN (Sequence Data)' },
    { id: 'transformer', name: 'Transformer (NLP)' },
    { id: 'rag', name: 'RAG (Knowledge Base)' },
    { id: 'svm', name: 'SVM (Classification)' },
    { id: 'pca', name: 'PCA (Dim. Reduction)' },
];

const ChatInterface = () => {
    const [messages, setMessages] = useState([
        { id: 1, text: "Hello! I'm your AI/ML Assistant. I can help you with Deep Learning, CNNs, RNNs, Transformers, and more. Which algorithm would you like to explore today?", sender: 'bot' }
    ]);
    const [input, setInput] = useState('');
    const [isTyping, setIsTyping] = useState(false);
    const [selectedModel, setSelectedModel] = useState('general');
    const messagesEndRef = useRef(null);

    const scrollToBottom = () => {
        messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
    };

    useEffect(() => {
        scrollToBottom();
    }, [messages]);

    const handleKeyPress = (e) => {
        if (e.key === 'Enter' && !e.shiftKey) {
            e.preventDefault();
            handleSend();
        }
    };

    const handleSend = async () => {
        if (!input.trim()) return;

        const userMessage = {
            id: messages.length + 1,
            text: input,
            sender: 'user'
        };

        setMessages(prev => [...prev, userMessage]);
        setInput('');
        setIsTyping(true);

        try {
            // Use environment variable for API URL, fallback to localhost if not set
            const API_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000';

            const response = await fetch(`${API_URL}/chat`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    message: userMessage.text,
                    model_id: selectedModel
                }),
            });

            if (!response.ok) {
                throw new Error('Network response was not ok');
            }

            const data = await response.json();

            const botMessage = {
                id: messages.length + 2,
                text: data.response,
                sender: 'bot',
                timestamp: data.timestamp // Optional: use if you want to display time
            };

            setMessages(prev => [...prev, botMessage]);
        } catch (error) {
            console.error('Error:', error);
            const errorMessage = {
                id: messages.length + 2,
                text: "⚠️ Connection Failed. If this is your first time running the app, the backend is likely still installing AI libraries (PyTorch, Transformers). Please check your terminal and wait for 'Uvicorn running' to appear.",
                sender: 'bot'
            };
            setMessages(prev => [...prev, errorMessage]);
        } finally {
            setIsTyping(false);
        }
    };

    return (
        <div className="chat-container">
            {/* Sidebar */}
            <div className="sidebar glass-panel">
                <div style={{ marginBottom: '2rem', padding: '0 0.5rem' }}>
                    <h2 className="gradient-text" style={{ fontSize: '1.5rem', fontWeight: 'bold' }}>NeuroChat</h2>
                    <p style={{ color: 'var(--text-muted)', fontSize: '0.875rem' }}>Advanced AI/ML Workspace</p>
                </div>

                <div style={{ flex: 1 }}>
                    <p style={{ color: 'var(--text-muted)', fontSize: '0.75rem', textTransform: 'uppercase', letterSpacing: '1px', marginBottom: '1rem', paddingLeft: '0.5rem' }}>
                        Select Model / Algorithm
                    </p>
                    {MODELS.map(model => (
                        <button
                            key={model.id}
                            className={`sidebar-btn ${selectedModel === model.id ? 'active' : ''}`}
                            onClick={() => setSelectedModel(model.id)}
                        >
                            {model.name}
                        </button>
                    ))}
                </div>

                <div style={{ borderTop: '1px solid var(--glass-border)', paddingTop: '1rem' }}>
                    <button className="sidebar-btn">
                        ⚙️ Settings
                    </button>
                </div>
            </div>

            {/* Main Chat Area */}
            <div className="main-chat">
                <div className="chat-header glass-panel">
                    <div>
                        <h3 style={{ fontWeight: 600 }}>{MODELS.find(m => m.id === selectedModel).name}</h3>
                        <span style={{ fontSize: '0.8rem', color: 'var(--accent-primary)', display: 'flex', alignItems: 'center', gap: '6px' }}>
                            <span style={{ width: '8px', height: '8px', background: '#22c55e', borderRadius: '50%' }}></span>
                            Online
                        </span>
                    </div>
                    <div>
                        {/* Header Actions */}
                    </div>
                </div>

                <div className="messages-area">
                    {messages.map(msg => (
                        <div key={msg.id} className={`message ${msg.sender}`}>
                            {msg.text}
                        </div>
                    ))}
                    {isTyping && (
                        <div className="message bot">
                            <div className="typing-indicator">
                                <div className="dot"></div>
                                <div className="dot"></div>
                                <div className="dot"></div>
                            </div>
                        </div>
                    )}
                    <div ref={messagesEndRef} />
                </div>

                <div className="input-area">
                    <div className="input-container">
                        <textarea
                            className="chat-input"
                            placeholder="Type your query about AI/ML..."
                            value={input}
                            onChange={(e) => setInput(e.target.value)}
                            onKeyDown={handleKeyPress}
                            rows={1}
                        />
                        <button
                            className="send-btn"
                            onClick={handleSend}
                            disabled={!input.trim() || isTyping}
                        >
                            <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                                <line x1="22" y1="2" x2="11" y2="13"></line>
                                <polygon points="22 2 15 22 11 13 2 9 22 2"></polygon>
                            </svg>
                        </button>
                    </div>
                    <div style={{ textAlign: 'center', marginTop: '0.5rem' }}>
                        <p style={{ fontSize: '0.75rem', color: 'var(--text-muted)' }}>
                            AI can make mistakes. Verify important information.
                        </p>
                    </div>
                </div>
            </div>
        </div>
    );
};

export default ChatInterface;
