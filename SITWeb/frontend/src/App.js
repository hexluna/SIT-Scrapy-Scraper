import logo from "./logo.svg";
import "./App.css";
import React, { useState } from "react";

function App() {
  const [prompt, setPrompt] = useState("");
  const [messages, setMessages] = useState([]);
  const [loading, setLoading] = useState(false);

  const handleSend = async () => {
    if (!prompt.trim()) return;
    setLoading(true);
    const userMessage = { sender: "You", text: prompt };
    setMessages((prev) => [...prev, userMessage]);

    try {
      const res = await fetch("http://localhost:5000/chat", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ prompt }),
      });
      const data = await res.json();
      const botMessage = { sender: "Bot", text: data.response || data.error };
      setMessages((prev) => [...prev, botMessage]);
    } catch (err) {
      setMessages((prev) => [
        ...prev,
        { sender: "Bot", text: "Error talking to server." },
      ]);
    }

    setPrompt("");
    setLoading(false);
  };

  return (
    <div
      style={{
        maxWidth: 600,
        margin: "0 auto",
        padding: 20,
        fontFamily: "Arial",
      }}
    >
      <h2>SIT Chatbot</h2>
      <div
        style={{
          border: "1px solid #ccc",
          padding: 10,
          height: 400,
          overflowY: "auto",
        }}
      >
        {messages.map((msg, i) => (
          <div key={i} style={{ margin: "10px 0" }}>
            <b>{msg.sender}:</b> <span>{msg.text}</span>
          </div>
        ))}
        {loading && (
          <p>
            <i>Bot is typing...</i>
          </p>
        )}
      </div>
      <input
        type="text"
        value={prompt}
        onChange={(e) => setPrompt(e.target.value)}
        placeholder="Ask something about SIT..."
        style={{ width: "80%", padding: 10, marginTop: 10 }}
        onKeyDown={(e) => e.key === "Enter" && handleSend()}
      />
      <button
        onClick={handleSend}
        disabled={loading}
        style={{ padding: "10px 20px", marginLeft: 10 }}
      >
        Send
      </button>
    </div>
  );
}

export default App;
