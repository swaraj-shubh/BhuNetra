import { createRoot } from "react-dom/client";
import "maplibre-gl/dist/maplibre-gl.css";
import "./app.css";
import App from "./App";

// M8: attach the analyst's API key (if one is set) to every same-origin API call.
const origFetch = window.fetch.bind(window);
window.fetch = (input: RequestInfo | URL, init: RequestInit = {}) => {
  let key = "";
  try { key = localStorage.getItem("apikey") ?? ""; } catch { /* storage blocked: no key */ }
  if (!key) return origFetch(input, init);
  const headers = new Headers(init.headers);
  headers.set("X-API-Key", key);
  return origFetch(input, { ...init, headers });
};

createRoot(document.getElementById("root")!).render(<App />);
