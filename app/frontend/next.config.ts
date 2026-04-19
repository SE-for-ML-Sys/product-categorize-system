import type { NextConfig } from "next";

const nextConfig: NextConfig = {
  reactStrictMode: true,
  // Produce a self-contained output directory for Docker image builds.
  // The standalone folder contains only the files needed to run the server.
  output: "standalone",
};

export default nextConfig;
