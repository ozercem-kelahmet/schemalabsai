/** @type {import('next').NextConfig} */
const nextConfig = {
  typescript: {
    ignoreBuildErrors: true,
  },
  images: {
    unoptimized: true,
  },
  async rewrites() {
    return {
      beforeFiles: [
        {
          source: '/api/organizations/:path*',
          destination: 'http://localhost:8080/api/organizations/:path*',
        },
        {
          source: '/api/organizations',
          destination: 'http://localhost:8080/api/organizations',
        },
        {
          source: '/api/admin/:path*',
          destination: 'http://localhost:8080/api/admin/:path*',
        },
        {
          source: '/api/playgrounds/:path*',
          destination: 'http://localhost:8080/api/playgrounds/:path*',
        },
        {
          source: '/api/playgrounds',
          destination: 'http://localhost:8080/api/playgrounds',
        },
        {
          source: '/api/auth/me',
          destination: 'http://localhost:8080/api/auth/me',
        },
      ],
    }
  },
}

export default nextConfig
