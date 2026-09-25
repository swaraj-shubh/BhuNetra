import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// Dev: `npm run dev` proxies the API (e.g. an ssh tunnel to the AWS box on :8000). Prod: FastAPI serves dist/.
const api = process.env.API ?? "http://127.0.0.1:8000";

export default defineConfig({
  plugins: [react()],
  server: { proxy: { "/search": api, "/chips": api, "/stats": api } },
});
