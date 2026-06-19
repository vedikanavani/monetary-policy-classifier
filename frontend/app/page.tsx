"use client";

import { useState } from "react";

export default function Home() {
  const [text, setText] = useState("");
  const [result, setResult] = useState("");
  const [loading, setLoading] = useState(false);

  const handlePredict = async () => {
    setLoading(true);
    setResult("");

    try {
      const res = await fetch("https://bank-nlp-project.onrender.com/predict", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({ text }),
      });

      if (!res.ok) {
        const errText = await res.text();
        throw new Error(errText);
      }

      const data = await res.json();
      setResult(data.prediction);
    } catch (err) {
      console.error("Request failed:", err);
      setResult("Error: check console");
    } finally {
      setLoading(false);
    }
  };

  return (
    <main className="container">
      <div className="card">

        <h1 className="title">
          Hawkish vs Dovish Classifier
        </h1>

        <p className="description">
          This tool analyzes uses a logistic regression model to classify statements as either
          hawkish or dovish based on the underlying monetary policy stance. A hawkish
          tone typically signals concerns about inflation and a tendency toward tighter
          monetary policy. A dovish tone reflects concern for economic growth and employment, often indicating a 
          preference for lower interest rates or accommodative policy. The model is trained on data available 
          from the GeorgiaTech Financial Services Innovation Lab. 
        </p>

        <div className="inputWrapper">
          <input
            className="input"
            value={text}
            onChange={(e) => setText(e.target.value)}
            placeholder="Type or paste a central bank statement..."
          />
        </div>

        <button className="button" onClick={handlePredict}>
          {loading ? "Analyzing..." : "Predict"}
        </button>

        {result && (
          <div className="resultBox">
            <h2>Result</h2>
            <p>{result}</p>
          </div>
        )}

      </div>

      <style jsx>{`
        .container {
          min-height: 100vh;
          display: flex;
          justify-content: center;
          align-items: center;
          background: #f5f0e6;
          font-family: Inter, system-ui, sans-serif;
          padding: 20px;
        }

        .card {
          width: 100%;
          max-width: 800px;
          text-align: center;
        }

        .title {
          font-size: 42px;
          font-weight: 600;
          margin-bottom: 12px;
          color: #1f1f1f;
        }

        .description {
          font-size: 14px;
          line-height: 1.6;
          color: #444;
          margin-bottom: 30px;
        }

        .inputWrapper {
          display: flex;
          justify-content: center;
          margin-bottom: 15px;
        }

        .input {
          width: 100%;
          padding: 16px 18px;
          font-size: 16px;
          border-radius: 14px;
          border: 1px solid #ddd;
          outline: none;
          background: white;
        }

        .input:focus {
          border-color: #999;
        }

        .button {
          padding: 12px 22px;
          font-size: 16px;
          border: none;
          border-radius: 12px;
          cursor: pointer;
          background: #1f1f1f;
          color: white;
          margin-top: 10px;
        }

        .button:hover {
          opacity: 0.9;
        }

        .resultBox {
          margin-top: 25px;
          padding: 18px;
          border-radius: 12px;
          background: white;
          border: 1px solid #ddd;
          color: #111;
        }

        .resultBox h2 {
          color: #1f1f1f;
          margin-bottom: 8px;
        }

        .resultBox p {
          color: #111;
          font-size: 18px;
          font-weight: 600;
        }
      `}</style>
    </main>
  );
}