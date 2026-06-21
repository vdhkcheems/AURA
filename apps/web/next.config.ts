import type { NextConfig } from "next";

const nextConfig: NextConfig = {
  // Qdrant configures an Undici dispatcher at runtime. Leaving it external
  // avoids conflicts with Next's instrumented server fetch implementation.
  serverExternalPackages: ["@qdrant/js-client-rest"],
};

export default nextConfig;
