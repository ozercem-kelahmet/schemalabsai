import { NextResponse } from "next/server"
import { cookies } from "next/headers"

const API_URL = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8080"

export async function POST(req: Request) {
  try {
    const cookieStore = await cookies()
    const session = cookieStore.get("session")
    
    if (!session) {
      return NextResponse.json({ error: "Unauthorized" }, { status: 401 })
    }

    const body = await req.json()

    const res = await fetch(`${API_URL}/api/models/finetuned/update`, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        Cookie: `session=${session.value}`,
      },
      body: JSON.stringify(body),
    })

    const data = await res.json()
    return NextResponse.json(data, { status: res.status })
  } catch (error) {
    console.error("Model update error:", error)
    return NextResponse.json({ error: "Failed to update model" }, { status: 500 })
  }
}
