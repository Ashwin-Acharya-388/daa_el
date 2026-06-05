/**
 * useWebSocket — Hook for WebSocket connections (batch progress).
 */
import { useState, useEffect, useRef, useCallback } from 'react';

const WS_BASE = import.meta.env.VITE_WS_URL || 'ws://localhost:8000';

export function useWebSocket(taskId) {
  const [messages, setMessages] = useState([]);
  const [lastMessage, setLastMessage] = useState(null);
  const [isConnected, setIsConnected] = useState(false);
  const wsRef = useRef(null);

  const connect = useCallback(() => {
    if (!taskId) return;

    const ws = new WebSocket(`${WS_BASE}/ws/batch/${taskId}`);
    wsRef.current = ws;

    ws.onopen = () => setIsConnected(true);

    ws.onmessage = (event) => {
      try {
        const data = JSON.parse(event.data);
        setLastMessage(data);
        setMessages((prev) => [...prev, data]);
      } catch {
        setLastMessage({ raw: event.data });
      }
    };

    ws.onclose = () => setIsConnected(false);
    ws.onerror = () => setIsConnected(false);
  }, [taskId]);

  const disconnect = useCallback(() => {
    if (wsRef.current) {
      wsRef.current.close();
      wsRef.current = null;
    }
  }, []);

  useEffect(() => {
    connect();
    return disconnect;
  }, [taskId]);

  return { messages, lastMessage, isConnected, connect, disconnect };
}

export default useWebSocket;
