import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// Dev server proxies /api to the gs-mem-server (default port 8088).
// Production build emits static assets into ./dist, which gs-mem-server
// embeds via rust-embed (wired in M4 server task).
export default defineConfig({
  plugins: [react()],
  build: { outDir: 'dist', emptyOutDir: false },
  server: {
    port: 5173,
    proxy: { '/api': 'http://127.0.0.1:8088' },
  },
})
