// src/App.jsx
import { useState, useRef, useEffect } from "react";
import ReactMarkdown from "react-markdown";

const API_URL = import.meta.env.VITE_API_URL ?? "http://localhost:8000/chat/";
const APP_VERSION = "1.0.1"; // Force cache refresh

// Quick action buttons for common diabetes queries
const QUICK_ACTIONS = [
  { text: "How's my blood sugar?", icon: "📊" },
  { text: "Predict my blood sugar", icon: "🔮" },
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

// Structured input component for blood sugar predictions
function BloodSugarPredictionForm({ onSubmit, onCancel, loading }) {
  const [timelineData, setTimelineData] = useState([]);
  const [errors, setErrors] = useState({});

  // Initialize timeline with 12 glucose readings (every 5 minutes, going backwards from current time)
  useEffect(() => {
    const now = new Date();
    const initialTimeline = [];
    
    // Default glucose values (from most recent to oldest)
    const defaultGlucoseValues = [117, 115, 112, 113, 108, 104, 111, 98, 94, 95, 92, 88];
    
    for (let i = 0; i < 12; i++) {
      const time = new Date(now.getTime() - (11 - i) * 5 * 60 * 1000); // 5 minute intervals
      initialTimeline.push({
        id: `glucose-${i}`,
        type: 'glucose',
        time: time,
        value: defaultGlucoseValues[i].toString(), // Set default value
        label: `Glucose Reading ${i + 1}`,
        timeLabel: time.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })
      });
    }
    
    setTimelineData(initialTimeline);
  }, []);

  const validateForm = () => {
    const newErrors = {};

    // Validate glucose readings
    const glucoseEntries = timelineData.filter(item => item.type === 'glucose');
    const glucoseErrors = glucoseEntries.map((entry, index) => {
      if (!entry.value.trim()) return `Glucose reading at ${entry.timeLabel} is required`;
      const num = parseFloat(entry.value);
      if (isNaN(num) || num < 0 || num > 600) return `Glucose reading at ${entry.timeLabel} must be between 0-600 mg/dL`;
      return null;
    });
    
    if (glucoseErrors.some(error => error)) {
      newErrors.glucose = glucoseErrors;
    }

    // Validate carbohydrates (optional entries)
    const carbEntries = timelineData.filter(item => item.type === 'carbohydrate');
    if (carbEntries.length > 0) {
      const carbErrors = carbEntries.map((entry, index) => {
        if (!entry.value.trim()) return `Carb amount at ${entry.timeLabel} is required`;
        const amount = parseFloat(entry.value);
        if (isNaN(amount) || amount < 0) return `Carb amount at ${entry.timeLabel} must be positive`;
        return null;
      });
      
      if (carbErrors.some(error => error)) {
        newErrors.carbohydrates = carbErrors;
      }
    }

    // Validate insulin (optional entries)
    const insulinEntries = timelineData.filter(item => item.type === 'insulin');
    if (insulinEntries.length > 0) {
      const insulinErrors = insulinEntries.map((entry, index) => {
        if (!entry.value.trim()) return `Insulin units at ${entry.timeLabel} is required`;
        const units = parseFloat(entry.value);
        if (isNaN(units) || units < 0) return `Insulin units at ${entry.timeLabel} must be positive`;
        return null;
      });
      
      if (insulinErrors.some(error => error)) {
        newErrors.insulin = insulinErrors;
      }
    }

    setErrors(newErrors);
    return Object.keys(newErrors).length === 0;
  };

  const handleSubmit = (e) => {
    e.preventDefault();
    if (validateForm()) {
      // Sort timeline data by time
      const sortedData = [...timelineData].sort((a, b) => a.time - b.time);
      
      const data = {
        glucoseReadings: sortedData
          .filter(item => item.type === 'glucose')
          .map(item => parseFloat(item.value)),
        carbohydrates: sortedData
          .filter(item => item.type === 'carbohydrate')
          .map(item => ({
            amount: parseFloat(item.value),
            timestamp: item.timeLabel
          })),
        insulinBolus: sortedData
          .filter(item => item.type === 'insulin')
          .map(item => ({
            units: parseFloat(item.value),
            timestamp: item.timeLabel
          }))
      };
      onSubmit(data);
    }
  };

  const addCarbohydrate = () => {
    const now = new Date();
    const newEntry = {
      id: `carb-${Date.now()}`,
      type: 'carbohydrate',
      time: now,
      value: '',
      label: 'Carbohydrates (g)',
      timeLabel: now.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })
    };
    
    // Insert at the appropriate position in timeline
    const newTimeline = [...timelineData, newEntry].sort((a, b) => a.time - b.time);
    setTimelineData(newTimeline);
  };

  const removeEntry = (id) => {
    setTimelineData(timelineData.filter(item => item.id !== id));
  };

  const addInsulin = () => {
    const now = new Date();
    const newEntry = {
      id: `insulin-${Date.now()}`,
      type: 'insulin',
      time: now,
      value: '',
      label: 'Insulin (units)',
      timeLabel: now.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })
    };
    
    // Insert at the appropriate position in timeline
    const newTimeline = [...timelineData, newEntry].sort((a, b) => a.time - b.time);
    setTimelineData(newTimeline);
  };

  const updateEntry = (id, value) => {
    setTimelineData(timelineData.map(item => 
      item.id === id ? { ...item, value } : item
    ));
  };

  return (
    <div className="bg-white border border-gray-200 rounded-xl p-6 mb-4">
      <h3 className="text-lg font-semibold text-gray-900 mb-4">Blood Sugar Prediction Timeline</h3>
      <div className="bg-blue-50 border border-blue-200 rounded-lg p-4 mb-6">
        <p className="text-sm text-blue-800">
          <strong>Instructions:</strong> Fill in your glucose readings for the past hour (every 5 minutes). Optionally add any carb intake or insulin boluses at their actual times. The timeline will automatically sort everything chronologically.
        </p>
      </div>
      
      {/* Add Entry Buttons */}
      <div className="flex gap-3 mb-6">
        <button
          type="button"
          onClick={addCarbohydrate}
          className="flex items-center gap-2 px-4 py-2 bg-green-100 text-green-700 rounded-lg hover:bg-green-200 transition-colors"
        >
          <span>🍽️</span>
          <span>Add Carbs</span>
        </button>
        <button
          type="button"
          onClick={addInsulin}
          className="flex items-center gap-2 px-4 py-2 bg-purple-100 text-purple-700 rounded-lg hover:bg-purple-200 transition-colors"
        >
          <span>💉</span>
          <span>Add Insulin</span>
        </button>
      </div>

      <form onSubmit={handleSubmit} className="space-y-4">
        {/* Timeline */}
        <div className="space-y-3">
          {timelineData
            .sort((a, b) => a.time - b.time)
            .map((entry) => (
              <div key={entry.id} className="flex items-center gap-4 p-3 border border-gray-200 rounded-lg">
                {/* Time */}
                <div className="w-20 text-sm font-medium text-gray-600">
                  {entry.timeLabel}
                </div>
                
                {/* Icon */}
                <div className="w-8 h-8 rounded-full flex items-center justify-center text-sm">
                  {entry.type === 'glucose' && <span className="text-blue-600">📊</span>}
                  {entry.type === 'carbohydrate' && <span className="text-green-600">🍽️</span>}
                  {entry.type === 'insulin' && <span className="text-purple-600">💉</span>}
                </div>
                
                {/* Label */}
                <div className="flex-1 text-sm font-medium text-gray-700">
                  {entry.label}
                </div>
                
                {/* Input */}
                <div className="w-32">
                  <input
                    type="number"
                    placeholder={entry.type === 'glucose' ? 'mg/dL' : entry.type === 'carbohydrate' ? 'g' : 'units'}
                    value={entry.value}
                    onChange={(e) => updateEntry(entry.id, e.target.value)}
                    className={`w-full px-3 py-2 border rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-500 text-sm ${
                      errors[entry.type] && errors[entry.type].some(error => error.includes(entry.timeLabel)) 
                        ? 'border-red-500' 
                        : 'border-gray-300'
                    }`}
                    min="0"
                    step="0.1"
                  />
                </div>
                
                {/* Remove button for non-glucose entries */}
                {entry.type !== 'glucose' && (
                  <button
                    type="button"
                    onClick={() => removeEntry(entry.id)}
                    className="px-2 py-1 text-red-500 hover:text-red-600 text-sm"
                  >
                    ✕
                  </button>
                )}
              </div>
            ))}
        </div>

        {/* Error Messages */}
        {Object.keys(errors).map(errorType => 
          errors[errorType] && errors[errorType].some(error => error) && (
            <div key={errorType} className="text-xs text-red-500 bg-red-50 p-3 rounded-lg">
              {errors[errorType].map((error, index) => error && <div key={index}>{error}</div>)}
            </div>
          )
        )}

        {/* Action Buttons */}
        <div className="flex gap-3 pt-4">
          <button
            type="submit"
            disabled={loading}
            className="flex-1 bg-blue-500 text-white py-3 px-4 rounded-lg hover:bg-blue-600 disabled:opacity-50 disabled:cursor-not-allowed font-medium"
          >
            {loading ? "Predicting..." : "Get Prediction"}
          </button>
          <button
            type="button"
            onClick={onCancel}
            disabled={loading}
            className="px-6 py-3 border border-gray-300 text-gray-700 rounded-lg hover:bg-gray-50 disabled:opacity-50"
          >
            Cancel
          </button>
        </div>
      </form>
    </div>
  );
}

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
  const [connectionStatus, setConnectionStatus] = useState("checking"); // "checking", "connected", "disconnected"
  const [showPredictionForm, setShowPredictionForm] = useState(false);
  const [predictionLoading, setPredictionLoading] = useState(false);
  const [conversationId, setConversationId] = useState(null); // Track current conversation
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

  // Check connection status on mount and periodically
  useEffect(() => {
      const checkConnection = async () => {
        try {
          const res = await fetch(API_URL.replace('/chat/', '/chat/model/status'), {
            method: "GET",
            mode: 'cors',
          });
          if (res.ok) {
            setConnectionStatus("connected");
          } else {
            setConnectionStatus("disconnected");
          }
        } catch (error) {
          setConnectionStatus("disconnected");
        }
      };

    checkConnection();
    const interval = setInterval(checkConnection, 30000); // Check every 30 seconds
    return () => clearInterval(interval);
  }, []);

  // Focus input on mount
  useEffect(() => {
    inputRef.current?.focus();
  }, []);

  const sendMessage = async (messageText) => {
    if (!messageText.trim() || loading) return;
    
    // Check if this is a blood sugar prediction request
    const isPredictionRequest = messageText.toLowerCase().includes('projected blood sugar') || 
                               messageText.toLowerCase().includes('blood sugar prediction') ||
                               messageText.toLowerCase().includes('predict') ||
                               messageText.toLowerCase().includes('forecast');
    
    if (isPredictionRequest) {
      setShowPredictionForm(true);
      const timestamp = new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
      const userMsg = { who: "user", text: messageText, timestamp };
      setHistory((h) => [...h, userMsg]);
      
      const assistantMsg = { 
        who: "assistant", 
        text: "I can help you predict your blood sugar levels! Please provide the following information:\n\n• **12 glucose readings** (every 5 minutes)\n• **Carbohydrates consumed** with timestamps\n• **Insulin bolus amounts** with timestamps\n\nThis will help me make an accurate prediction using our AI model.",
        timestamp: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })
      };
      setHistory((h) => [...h, assistantMsg]);
      return;
    }
    
    const timestamp = new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
    const userMsg = { who: "user", text: messageText, timestamp };
    
    setHistory((h) => [...h, userMsg]);
    setInput("");
    setLoading(true);
    
    try {
      console.log('Sending message to:', API_URL);
      console.log('Message data:', { 
        subject_id: subjectId, 
        text: messageText, 
        conversation_id: conversationId 
      });
      
      const res = await fetch(API_URL, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ 
          subject_id: subjectId, 
          text: messageText, 
          conversation_id: conversationId 
        }),
        mode: 'cors',
      });
      
      console.log('Response status:', res.status, res.ok);
      
      if (!res.ok) {
        throw new Error(`HTTP error! status: ${res.status}`);
      }
      
      const data = await res.json();
      console.log('Response data:', data);
      
      // Update conversation ID if this is a new conversation
      if (data.conversation_id && !conversationId) {
        setConversationId(data.conversation_id);
        console.log('New conversation started:', data.conversation_id);
      }
      
      const assistantMsg = { 
        who: "assistant", 
        text: data.reply, 
        timestamp: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })
      };
      setHistory((h) => [...h, assistantMsg]);
    } catch (e) {
      console.error('Error sending message:', e);
      let errorText = "⚠️ I'm having trouble connecting right now. Please try again in a moment.";
      
      // Provide more specific error messages for debugging
      if (e.name === 'TypeError' && e.message.includes('fetch')) {
        errorText = "⚠️ Network error: Unable to connect to the server. Please check if the backend is running.";
      } else if (e.message.includes('HTTP error')) {
        errorText = `⚠️ Server error: ${e.message}`;
      }
      
      const errorMsg = { 
        who: "assistant", 
        text: errorText, 
        timestamp: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })
      };
      setHistory((h) => [...h, errorMsg]);
    } finally {
      setLoading(false);
    }
  };

  const handleStructuredPrediction = async (predictionData) => {
    setPredictionLoading(true);
    
    try {
      console.log('Sending structured prediction data:', predictionData);
      
      const res = await fetch(API_URL + 'predict', {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ 
          subject_id: subjectId, 
          ...predictionData 
        }),
        mode: 'cors',
      });
      
      if (!res.ok) {
        throw new Error(`HTTP error! status: ${res.status}`);
      }
      
      const data = await res.json();
      console.log('Prediction response:', data);
      
      const predictionText = data.prediction_text || `Based on your data, I predict your blood sugar will be **${data.prediction?.toFixed(1)} mg/dL** in the next time period.`;
      
      // Add the prediction to the conversation context by sending it through the chat endpoint
      const contextRes = await fetch(API_URL, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ 
          subject_id: subjectId, 
          text: `[PREDICTION RESULT] ${predictionText}`,
          conversation_id: conversationId 
        }),
        mode: 'cors',
      });
      
      if (contextRes.ok) {
        const contextData = await contextRes.json();
        // Update conversation ID if this is a new conversation
        if (contextData.conversation_id && !conversationId) {
          setConversationId(contextData.conversation_id);
        }
      }
      
      const assistantMsg = { 
        who: "assistant", 
        text: predictionText,
        timestamp: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })
      };
      setHistory((h) => [...h, assistantMsg]);
      
      setShowPredictionForm(false);
    } catch (e) {
      console.error('Error making prediction:', e);
      const errorMsg = { 
        who: "assistant", 
        text: "⚠️ I'm having trouble making the prediction right now. Please try again in a moment.",
        timestamp: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })
      };
      setHistory((h) => [...h, errorMsg]);
    } finally {
      setPredictionLoading(false);
    }
  };

  const handleCancelPrediction = () => {
    setShowPredictionForm(false);
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

  const startNewConversation = () => {
    setHistory([]);
    setConversationId(null);
    setShowPredictionForm(false);
  };

  const handleEmergency = () => {
    // In a real app, this would trigger emergency protocols
    alert("Emergency protocols activated. Contacting emergency services...");
  };

  // Expose connection test function globally for debugging
  useEffect(() => {
    window.testConnection = async () => {
      console.log('Manual connection test triggered');
      try {
        const response = await fetch('http://localhost:8000/ping', {
          method: 'GET',
          headers: { 'Content-Type': 'application/json' },
          mode: 'cors',
          cache: 'no-cache',
        });
        console.log('Manual test response:', response.status, response.ok);
        if (response.ok) {
          const data = await response.json();
          console.log('Manual test data:', data);
          setConnectionStatus("connected");
          return true;
        } else {
          console.error('Manual test failed:', response.status);
          setConnectionStatus("disconnected");
          return false;
        }
      } catch (error) {
        console.error('Manual test error:', error);
        setConnectionStatus("disconnected");
        return false;
      }
    };
  }, []);



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
              {conversationId && (
                <p className="text-xs text-gray-400 mt-1">
                  Conversation: {conversationId.slice(0, 8)}...
                </p>
              )}
            </div>
            <div className="flex items-center gap-2">
              <button
                onClick={startNewConversation}
                className="px-3 py-1.5 text-sm font-medium bg-gray-100 text-gray-700 rounded-md hover:bg-gray-200 transition-colors"
              >
                New Chat
              </button>
              <div className={`w-2 h-2 rounded-full ${
                connectionStatus === "connected" ? "bg-green-500" : 
                connectionStatus === "checking" ? "bg-yellow-500" : "bg-red-500"
              } ${connectionStatus === "checking" ? "animate-pulse" : ""}`}></div>
              <span className="text-sm text-gray-500">
                {connectionStatus === "connected" ? "Online" : 
                 connectionStatus === "checking" ? "Checking..." : "Offline"}
              </span>
              <button 
                onClick={() => {
                  setConnectionStatus("checking");
                  setTimeout(() => {
                    const checkConnection = async () => {
                      try {
                        const response = await fetch(API_URL.replace('/chat/', '/ping'));
                        if (response.ok) {
                          setConnectionStatus("connected");
                        } else {
                          setConnectionStatus("disconnected");
                        }
                      } catch (error) {
                        console.error('Manual connection check failed:', error);
                        setConnectionStatus("disconnected");
                      }
                    };
                    checkConnection();
                  }, 100);
                }}
                className="text-xs text-blue-500 hover:text-blue-600 underline"
              >
                Retry
              </button>
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
          
          {showPredictionForm && (
            <div className="mb-4">
              <BloodSugarPredictionForm
                onSubmit={handleStructuredPrediction}
                onCancel={handleCancelPrediction}
                loading={predictionLoading}
              />
            </div>
          )}
          
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
