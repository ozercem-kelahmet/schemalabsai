import { NextResponse } from "next/server"
import { cookies } from "next/headers"

const API_URL = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8080"

export async function GET() {
  try {
    const cookieStore = await cookies()
    const session = cookieStore.get("session")
    
    if (!session) {
      return NextResponse.json({ queries: [] })
    }

    const url = API_URL + "/api/queries"
    const res = await fetch(url, {
      headers: {
        Cookie: "session=" + session.value,
      },
    })

    if (!res.ok) {
      return NextResponse.json({ queries: [] })
    }

    const data = await res.json()
    return NextResponse.json(data)
  } catch (error) {
    console.error("Queries fetch error:", error)
    return NextResponse.json({ queries: [] })
  }
}
