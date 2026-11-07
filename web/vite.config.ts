import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";
import tailwindcss from "@tailwindcss/vite";

// typed locally: the app itself has no Node types
declare const process: { env: Record<string, string | undefined> };
const backend = process.env.DOCCHECK_BACKEND ?? "http://127.0.0.1:8000";

export default defineConfig({
  plugins: [react(), tailwindcss()],
  server: {
    port: 5173,
    proxy: { "/api": { target: backend, changeOrigin: true } },
  },
  build: { chunkSizeWarningLimit: 1500 },
});
