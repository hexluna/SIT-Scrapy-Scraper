import logo from "./logo.svg";
import "./App.css";
import React, { useState } from "react";

function App() {
  const [prompt, setPrompt] = useState("");
  const [messages, setMessages] = useState([]);
  const [loading, setLoading] = useState(false);
  const [uploading, setUploading] = useState(false);
  const [showGreeting, setShowGreeting] = useState(true);

  const handleSend = async () => {
  if (!prompt.trim()) return;
  const userMessage = { sender: "You", text: prompt };
  setMessages((prev) => [...prev, userMessage]);
  setLoading(true);
  setPrompt("");

  const res = await fetch("http://localhost:5000/chat", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ prompt }),
  });

  const reader = res.body.getReader();
  const decoder = new TextDecoder("utf-8");

  let botText = "";
  setMessages((prev) => [...prev, { sender: "Bot", text: "" }]); // placeholder

  while (true) {
    const { done, value } = await reader.read();
    if (done) break;

    const chunk = decoder.decode(value, { stream: true });
    botText += chunk;

    setMessages((prev) => {
      const updated = [...prev];
      updated[updated.length - 1] = { sender: "Bot", text: botText };
      return updated;
    });
  }

  setLoading(false);
};
  const handleFileUpload = async (file) => {
    if (!file || file.type !== "application/pdf") {
      alert("Please upload a valid PDF file.");
      return;
    }

    setUploading(true);
    const formData = new FormData();
    formData.append("file", file);

    const res = await fetch("http://localhost:5000/upload", {
      method: "POST",
      body: formData,
    });

    const reader = res.body.getReader();
    const decoder = new TextDecoder("utf-8");

    let botText = "";
    setMessages((prev) => [...prev, {sender: "Bot", text: ""}]);

    while (true) {
      const {done, value} = await reader.read();
      if (done) break;

      const chunk = decoder.decode(value, {stream: true});
      botText += chunk;

      setMessages((prev) => {
        const updated = [...prev];
        updated[updated.length - 1] = {sender: "Bot", text: botText};
        return updated;
      });
    }
  }

  const handleDrop = (e) => {
    e.preventDefault();
    const file = e.dataTransfer.files[0];
    handleFileUpload(file);
  };

  const handleDragOver = (e) => {
    e.preventDefault();
  };

  const renderMessageWithLinks = (msg) => {
    if (msg.sender !== "Bot")
      return <span style={{ whiteSpace: "pre-line" }}>{msg.text}</span>;

    const regex = /\[([^\]]+)\]\((https?:\/\/[^)]+)\)|<((https?:\/\/[^>]+))>/g;
    const parts = [];
    let lastIndex = 0;
    let match;

    while ((match = regex.exec(msg.text)) !== null) {
      if (match.index > lastIndex) {
        parts.push(
          <span key={lastIndex}>{msg.text.slice(lastIndex, match.index)}</span>
        );
      }

      if (match[1] && match[2]) {
        parts.push(
          <a
            key={match.index}
            href={match[2]}
            target="_blank"
            rel="noopener noreferrer"
          >
            {match[1]}
          </a>
        );
      } else if (match[3]) {
        parts.push(
          <a
            key={match.index}
            href={match[3]}
            target="_blank"
            rel="noopener noreferrer"
          >
            {match[3]}
          </a>
        );
      }
      lastIndex = regex.lastIndex;
    }

    if (lastIndex < msg.text.length) {
      parts.push(<span key={lastIndex}>{msg.text.slice(lastIndex)}</span>);
    }

    return <span style={{ whiteSpace: "pre-line" }}>{parts}</span>;
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
      {/* Website Header */}
      <header
        style={{
          display: "flex",
          alignItems: "center",
          marginBottom: 20,
        }}
      >
        <a
          href="#"
          style={{
            display: "flex",
            alignItems: "center",
            textDecoration: "none",
          }}
        >
          <img
            src={logo}
            alt="Logo"
            style={{ width: 40, height: 40, marginRight: 10 }}
          />
          <span style={{ fontSize: 18, fontWeight: "bold", color: "#333" }}>
            SIT Portal
          </span>
        </a>
      </header>

      {showGreeting && (
        <div
          style={{
            backgroundColor: "#e6f2ff",
            border: "1px solid #3399ff",
            borderRadius: 12,
            padding: 25,
            marginBottom: 25,
            textAlign: "left",
            boxShadow: "0 4px 8px rgba(0, 0, 0, 0.1)",
            position: "relative",
          }}
        >
          <button
            onClick={() => setShowGreeting(false)}
            style={{
              position: "absolute",
              top: 10,
              right: 15,
              background: "transparent",
              border: "none",
              fontSize: 20,
              fontWeight: "bold",
              color: "#666",
              cursor: "pointer",
            }}
          >
            &times;
          </button>
          <h2 style={{ color: "#003366", marginBottom: 10 }}>
            👋 Hello and Welcome!
          </h2>
          <p style={{ margin: 0, fontSize: 16 }}>
            This is the <strong>SIT Chatbot</strong>, your friendly assistant to
            all things SIT.
            <br />
            Ask about programs, admissions, deadlines, or anything else related
            to SIT.
          </p>
        </div>
      )}
      <h2>SIT Chatbot</h2>
      {/* Drag & Drop PDF */}
      <div
        onDrop={handleDrop}
        onDragOver={handleDragOver}
        style={{
          border: "2px dashed #999",
          borderRadius: 8,
          padding: 20,
          textAlign: "center",
          marginBottom: 10,
        }}
      >
        {uploading ? "Uploading and processing PDF..." : "Drag and drop a PDF file here"}
      </div>

      {/* Chat messages */}
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
            <b>{msg.sender}:</b> <span style={{ whiteSpace: 'pre-line' }}>{msg.text}</span>
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
