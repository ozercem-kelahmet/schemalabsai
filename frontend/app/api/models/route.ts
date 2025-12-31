import { NextResponse } from "next/server"
import { cookies } from "next/headers"

const API_URL = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8080"

export async function GET() {
  try {
    const cookieStore = await cookies()
    const session = cookieStore.get("session")
    
    if (!session) {
      return NextResponse.json({ models: [] })
    }

    const res = await fetch(`${API_URL}/api/models/finetuned`, {
      headers: {
        Cookie: `session=${session.value}`,
      },
    })

    if (!res.ok) {
      return NextResponse.json({ models: [] })
    }

    const data = await res.json()
    return NextResponse.json(data)
  } catch (error) {
    console.error("Models fetch error:", error)
    return NextResponse.json({ models: [] })
  }
}
