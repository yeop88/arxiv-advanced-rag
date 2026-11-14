import { defineConfig } from "vitest/config";
import react from "@vitejs/plugin-react";

// 개발: npm run dev → http://localhost:5173 (API 는 FastAPI 서버 8000 으로 넘김)
// 배포: npm run build → dist/ 를 FastAPI 가 / 에서 제공
export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    proxy: { "/api": { target: "http://127.0.0.1:8000", changeOrigin: true } },
  },
  // 로컬에서 쓰는 앱이라 한 묶음(약 190KB gzip)으로 둔다. 수동 분할(manualChunks)은 묶음 간 순환 참조로 화면이 깨졌다.
  build: { chunkSizeWarningLimit: 800 },
  test: { environment: "node" },
});
