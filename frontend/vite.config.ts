import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// In development the API runs separately (python -m api --no-browser) on port 8000.
// In production the built app is served by that same API, so /api is same-origin.
export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    proxy: { "/api": "http://127.0.0.1:8000" },
  },
  build: { outDir: "dist", sourcemap: false, chunkSizeWarningLimit: 800 },
});
