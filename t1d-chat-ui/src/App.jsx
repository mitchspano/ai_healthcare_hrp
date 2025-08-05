// src/App.jsx
import { useState, useRef, useEffect } from "react";

const API_URL = import.meta.env.VITE_API_URL ?? "http://localhost:8000/chat";

function Bubble({ who, text }) {
  // blue for user, gray for assistant
  const style =
    who === "user"
      ? "bg-blue-600 text-white ml-auto"
      : "bg-gray-200 text-gray-900";
  return (
    <div className={`rounded-lg px-3 py-2 max-w-xs ${style}`}>
      {text}
    </div>
  );
}

export default function App() {
  const [history, setHistory] = useState([]);
  const [input, setInput] = useState("");
  const [loading, setLoading] = useState(false);
  const endRef = useRef(null);

  // auto-scroll to newest message
  useEffect(() => {
    endRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [history]);

  const send = async () => {
    if (!input.trim() || loading) return;
    const userMsg = { who: "user", text: input };
    setHistory((h) => [...h, userMsg]);
    setInput("");
    setLoading(true);
    try {
      const res = await fetch(API_URL, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ subject_id: "Subject 9", text: input }),
      });
      const data = await res.json();
      setHistory((h) => [...h, { who: "assistant", text: data.reply }]);
    } catch (e) {
      setHistory((h) => [
        ...h,
        { who: "assistant", text: "⚠️ backend error, try again." },
      ]);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="h-screen flex flex-col items-center bg-gray-50">
      <header className="text-2xl font-semibold my-4">T1D Chat MVP</header>

      {/* Chat log */}
      <div className="flex-1 w-full max-w-lg overflow-y-auto px-2 space-y-2">
        {history.map((m, i) => (
          <Bubble key={i} {...m} />
        ))}
        <div ref={endRef} />
      </div>

      {/* Input bar */}
      <div className="w-full max-w-lg mb-4 flex gap-2 px-2">
        <input
          className="flex-1 border rounded-lg p-2"
          placeholder="Ask me about your blood sugar…"
          value={input}
          onChange={(e) => setInput(e.target.value)}
          onKeyDown={(e) => e.key === "Enter" && send()}
          disabled={loading}
        />
        <button
          className="bg-blue-600 text-white px-4 py-2 rounded-lg disabled:opacity-50"
          onClick={send}
          disabled={loading}
        >
          {loading ? "…" : "Send"}
        </button>
      </div>
    </div>
  );
}
