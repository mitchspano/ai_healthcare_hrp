// src/App.jsx
import { useState, useRef, useEffect } from "react";
import ReactMarkdown from "react-markdown";

const API_URL = import.meta.env.VITE_API_URL ?? "http://localhost:8000/chat";

// Quick action buttons for common diabetes queries
const QUICK_ACTIONS = [
  { text: "How's my blood sugar?", icon: "📊" },
  { text: "What should I eat?", icon: "🍽️" },
  { text: "Exercise recommendations", icon: "🏃‍♂️" },
  { text: "Medication reminder", icon: "💊" },
  { text: "Emergency symptoms", icon: "⚠️" },
  { text: "Weekly summary", icon: "📈" },
];

// Mock health metrics - in a real app, this would come from your backend
const MOCK_METRICS = {
  currentBg: 142,
  trend: "↗️", // rising
  lastReading: "2 hours ago",
  average: 138,
  range: "70-180",
  insulinOnBoard: 2.3,
  lastMeal: "3 hours ago"
};

function HealthMetricsCard() {
  const [metrics] = useState(MOCK_METRICS);
  
  return (
    <div className="bg-white rounded-xl border border-gray-200 p-4 mb-6">
      <h3 className="text-sm font-semibold text-gray-700 mb-3">Current Status</h3>
      <div className="grid grid-cols-2 gap-4">
        <div className="text-center">
          <div className="text-2xl font-bold text-gray-900">{metrics.currentBg}</div>
          <div className="text-sm text-gray-500">mg/dL</div>
          <div className="text-lg">{metrics.trend}</div>
        </div>
        <div className="space-y-2">
          <div className="flex justify-between text-sm">
            <span className="text-gray-500">Average:</span>
            <span className="font-medium">{metrics.average} mg/dL</span>
          </div>
          <div className="flex justify-between text-sm">
            <span className="text-gray-500">Range:</span>
            <span className="font-medium">{metrics.range}</span>
          </div>
          <div className="flex justify-between text-sm">
            <span className="text-gray-500">IOB:</span>
            <span className="font-medium">{metrics.insulinOnBoard} units</span>
          </div>
        </div>
      </div>
      <div className="mt-3 pt-3 border-t border-gray-100">
        <div className="flex justify-between text-xs text-gray-500">
          <span>Last reading: {metrics.lastReading}</span>
          <span>Last meal: {metrics.lastMeal}</span>
        </div>
      </div>
    </div>
  );
}

function MessageBubble({ who, text, timestamp }) {
  const isUser = who === "user";
  
  return (
    <div className={`flex ${isUser ? 'justify-end' : 'justify-start'} mb-4`}>
      <div className={`flex ${isUser ? 'flex-row-reverse' : 'flex-row'} items-end max-w-[80%] gap-2`}>
        {/* Avatar */}
        <div className={`w-8 h-8 rounded-full flex items-center justify-center text-sm font-semibold flex-shrink-0 ${
          isUser 
            ? 'bg-blue-500 text-white' 
            : 'bg-green-500 text-white'
        }`}>
          {isUser ? '👤' : '🤖'}
        </div>
        
        {/* Message bubble */}
        <div className={`rounded-2xl px-4 py-3 shadow-sm ${
          isUser
            ? 'bg-blue-500 text-white rounded-br-md'
            : 'bg-white text-gray-800 rounded-bl-md border border-gray-200'
        }`}>
          {isUser ? (
            <p className="text-sm leading-relaxed whitespace-pre-wrap">{text}</p>
          ) : (
            <div className="text-sm leading-relaxed chat-markdown">
              <ReactMarkdown
                components={{
                  // Custom styling for markdown elements
                  h1: ({ children }) => <h1 className="text-lg font-bold mb-2 text-gray-900">{children}</h1>,
                  h2: ({ children }) => <h2 className="text-base font-semibold mb-2 text-gray-900">{children}</h2>,
                  h3: ({ children }) => <h3 className="text-sm font-semibold mb-1 text-gray-900">{children}</h3>,
                  p: ({ children }) => <p className="mb-2 last:mb-0">{children}</p>,
                  ul: ({ children }) => <ul className="list-disc list-inside mb-2 space-y-1">{children}</ul>,
                  ol: ({ children }) => <ol className="list-decimal list-inside mb-2 space-y-1">{children}</ol>,
                  li: ({ children }) => <li className="text-sm">{children}</li>,
                  strong: ({ children }) => <strong className="font-semibold text-gray-900">{children}</strong>,
                  em: ({ children }) => <em className="italic">{children}</em>,
                  code: ({ children }) => <code className="bg-gray-100 px-1 py-0.5 rounded text-xs font-mono">{children}</code>,
                  blockquote: ({ children }) => <blockquote className="border-l-4 border-blue-200 pl-3 italic text-gray-600">{children}</blockquote>,
                  hr: () => <hr className="my-3 border-gray-200" />,
                }}
              >
                {text}
              </ReactMarkdown>
            </div>
          )}
          {timestamp && (
            <p className={`text-xs mt-1 ${
              isUser ? 'text-blue-100' : 'text-gray-500'
            }`}>
              {timestamp}
            </p>
          )}
        </div>
      </div>
    </div>
  );
}

function QuickActionButton({ action, onClick, disabled }) {
  return (
    <button
      onClick={() => onClick(action.text)}
      disabled={disabled}
      className="flex items-center gap-2 px-4 py-2 bg-white border border-gray-200 rounded-xl hover:bg-gray-50 hover:border-gray-300 transition-all duration-200 disabled:opacity-50 disabled:cursor-not-allowed shadow-sm"
    >
      <span className="text-lg">{action.icon}</span>
      <span className="text-sm font-medium text-gray-700">{action.text}</span>
    </button>
  );
}

function LoadingIndicator() {
  return (
    <div className="flex justify-start mb-4">
      <div className="flex items-end gap-2">
        <div className="w-8 h-8 rounded-full bg-green-500 text-white flex items-center justify-center text-sm font-semibold">
          🤖
        </div>
        <div className="bg-white border border-gray-200 rounded-2xl rounded-bl-md px-4 py-3 shadow-sm">
          <div className="flex space-x-1">
            <div className="w-2 h-2 bg-gray-400 rounded-full animate-bounce"></div>
            <div className="w-2 h-2 bg-gray-400 rounded-full animate-bounce" style={{ animationDelay: '0.1s' }}></div>
            <div className="w-2 h-2 bg-gray-400 rounded-full animate-bounce" style={{ animationDelay: '0.2s' }}></div>
          </div>
        </div>
      </div>
    </div>
  );
}

function EmergencyButton({ onClick }) {
  return (
    <button 
      onClick={onClick}
      className="w-full bg-red-500 hover:bg-red-600 text-white font-semibold py-3 px-4 rounded-xl transition-colors duration-200 flex items-center justify-center gap-2"
    >
      <span className="text-lg">🚨</span>
      <span>Emergency Help</span>
    </button>
  );
}



export default function App() {
  const [history, setHistory] = useState([]);
  const [input, setInput] = useState("");
  const [loading, setLoading] = useState(false);
  const [subjectId] = useState("Subject 9"); // In a real app, this would come from auth
  const [showMetrics, setShowMetrics] = useState(true);
  const inputRef = useRef(null);
  const messagesContainerRef = useRef(null);

  // Auto-scroll to newest message within the chat container only
  useEffect(() => {
    const timer = setTimeout(() => {
      if (messagesContainerRef.current) {
        messagesContainerRef.current.scrollTop = messagesContainerRef.current.scrollHeight;
      }
    }, 100);
    return () => clearTimeout(timer);
  }, [history, loading]);

  // Focus input on mount
  useEffect(() => {
    inputRef.current?.focus();
  }, []);

  const sendMessage = async (messageText) => {
    if (!messageText.trim() || loading) return;
    
    const timestamp = new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
    const userMsg = { who: "user", text: messageText, timestamp };
    
    setHistory((h) => [...h, userMsg]);
    setInput("");
    setLoading(true);
    
    try {
      const res = await fetch(API_URL, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ subject_id: subjectId, text: messageText }),
      });
      
      if (!res.ok) {
        throw new Error(`HTTP error! status: ${res.status}`);
      }
      
      const data = await res.json();
      const assistantMsg = { 
        who: "assistant", 
        text: data.reply, 
        timestamp: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })
      };
      setHistory((h) => [...h, assistantMsg]);
    } catch (e) {
      console.error('Error sending message:', e);
      const errorMsg = { 
        who: "assistant", 
        text: "⚠️ I'm having trouble connecting right now. Please try again in a moment.", 
        timestamp: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })
      };
      setHistory((h) => [...h, errorMsg]);
    } finally {
      setLoading(false);
    }
  };

  const handleSend = () => {
    sendMessage(input);
  };

  const handleQuickAction = (actionText) => {
    sendMessage(actionText);
  };

  const handleKeyDown = (e) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      handleSend();
    }
  };

  const handleEmergency = () => {
    // In a real app, this would trigger emergency protocols
    alert("Emergency protocols activated. Contacting emergency services...");
  };



  return (
    <div className="h-screen flex bg-gray-50">
      {/* Sidebar */}
      <div className="w-80 bg-white border-r border-gray-200 flex flex-col">
        {/* Header */}
        <div className="p-6 border-b border-gray-200">
          <div className="flex items-center gap-3">
            <div className="w-10 h-10 bg-gradient-to-br from-blue-500 to-purple-600 rounded-xl flex items-center justify-center">
              <span className="text-white text-lg font-bold">T1D</span>
            </div>
            <div>
              <h1 className="text-xl font-bold text-gray-900">Diabetes Assistant</h1>
              <p className="text-sm text-gray-500">Your health companion</p>
            </div>
          </div>
        </div>

        {/* User Info */}
        <div className="p-6 border-b border-gray-200">
          <div className="flex items-center gap-3">
            <div className="w-12 h-12 bg-blue-100 rounded-full flex items-center justify-center">
              <span className="text-blue-600 text-lg">👤</span>
            </div>
            <div>
              <p className="font-medium text-gray-900">{subjectId}</p>
              <p className="text-sm text-gray-500">Type 1 Diabetes</p>
            </div>
          </div>
        </div>

        {/* Health Metrics */}
        {showMetrics && (
          <div className="px-6 py-4 border-b border-gray-200">
            <HealthMetricsCard />
          </div>
        )}

        {/* Quick Actions */}
        <div className="flex-1 p-6">
          <div className="flex items-center justify-between mb-4">
            <h3 className="text-sm font-semibold text-gray-700">Quick Actions</h3>
            <button
              onClick={() => setShowMetrics(!showMetrics)}
              className="text-xs text-blue-500 hover:text-blue-600"
            >
              {showMetrics ? 'Hide' : 'Show'} Metrics
            </button>
          </div>
          <div className="space-y-3">
            {QUICK_ACTIONS.map((action, index) => (
              <QuickActionButton
                key={index}
                action={action}
                onClick={handleQuickAction}
                disabled={loading}
              />
            ))}
          </div>
        </div>

        {/* Emergency Button */}
        <div className="p-6 border-t border-gray-200">
          <EmergencyButton onClick={handleEmergency} />
        </div>

        {/* Footer */}
        <div className="p-6 border-t border-gray-200">
          <div className="text-xs text-gray-500 text-center">
            <p>Powered by AI Healthcare</p>
            <p className="mt-1">Always consult your doctor</p>
          </div>
        </div>
      </div>

      {/* Main Chat Area */}
      <div className="flex-1 flex flex-col">
        {/* Chat Header */}
        <div className="bg-white border-b border-gray-200 px-6 py-4">
          <div className="flex items-center justify-between">
            <div>
              <h2 className="text-lg font-semibold text-gray-900">Chat</h2>
              <p className="text-sm text-gray-500">Ask me anything about your diabetes management</p>
            </div>
            <div className="flex items-center gap-2">
              <div className="w-2 h-2 bg-green-500 rounded-full pulse"></div>
              <span className="text-sm text-gray-500">Online</span>
            </div>
          </div>
        </div>

        {/* Messages */}
        <div ref={messagesContainerRef} className="flex-1 overflow-y-auto p-6">
          {history.length === 0 && (
            <div className="text-center py-12">
              <div className="w-16 h-16 bg-gradient-to-br from-blue-500 to-purple-600 rounded-full flex items-center justify-center mx-auto mb-4">
                <span className="text-white text-2xl">💬</span>
              </div>
              <h3 className="text-lg font-semibold text-gray-900 mb-2">Welcome to your Diabetes Assistant!</h3>
              <p className="text-gray-500 mb-6">I'm here to help you manage your diabetes. Ask me about your blood sugar, diet, exercise, or anything else.</p>
              <div className="grid grid-cols-2 gap-3 max-w-md mx-auto">
                {QUICK_ACTIONS.slice(0, 4).map((action, index) => (
                  <QuickActionButton
                    key={index}
                    action={action}
                    onClick={handleQuickAction}
                    disabled={loading}
                  />
                ))}
              </div>
            </div>
          )}
          
          {history.map((message, i) => (
            <MessageBubble key={i} {...message} />
          ))}
          
          {loading && <LoadingIndicator />}
        </div>

        {/* Input Area */}
        <div className="bg-white border-t border-gray-200 p-6">
          <div className="flex gap-3">
            <div className="flex-1 relative">
              <input
                ref={inputRef}
                type="text"
                className="w-full px-4 py-3 border border-gray-300 rounded-xl focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent pr-12"
                placeholder="Type your message here..."
                value={input}
                onChange={(e) => setInput(e.target.value)}
                onKeyDown={handleKeyDown}
                disabled={loading}
              />
              <button
                onClick={handleSend}
                disabled={!input.trim() || loading}
                className="absolute right-2 top-1/2 transform -translate-y-1/2 w-8 h-8 bg-blue-500 text-white rounded-lg flex items-center justify-center hover:bg-blue-600 disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
              >
                <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 19l9 2-9-18-9 18 9-2zm0 0v-8" />
                </svg>
              </button>
            </div>
          </div>
          <p className="text-xs text-gray-500 mt-2 text-center">
            Press Enter to send • Shift+Enter for new line
          </p>
        </div>
      </div>
    </div>
  );
}
