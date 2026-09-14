import { defineConfig } from 'vite'
import vue from '@vitejs/plugin-vue'

// During dev, everything hitting /api is proxied to the Python backend.
// In production, serve the built assets behind the same origin as the API
// (or set a build-time env var and rewrite the base URL in src/lib/api.js).
export default defineConfig({
  plugins: [vue()],
  server: {
    port: 5173,
    proxy: {
      '/api': {
        target: 'http://localhost:8000',
        changeOrigin: true,
        rewrite: (path) => path.replace(/^\/api/, ''),
      },
    },
  },
})
