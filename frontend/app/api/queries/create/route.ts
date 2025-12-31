import { NextResponse } from "next/server"
import { cookies } from "next/headers"

const API_URL = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8080"

export async function POST(request: Request) {
  try {
    const cookieStore = await cookies()
    const session = cookieStore.get("session")
    
    if (!session) {
      return NextResponse.json({ error: "Unauthorized" }, { status: 401 })
    }

    const body = await request.json()
    console.log("Creating query with body:", body)

    const url = API_URL + "/api/queries/create"
    const res = await fetch(url, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        Cookie: "session=" + session.value,
      },
      body: JSON.stringify(body),
    })

    if (!res.ok) {
      const errText = await res.text()
      console.error("Backend error:", errText)
      return NextResponse.json({ error: "Failed to create query" }, { status: res.status })
    }

    const data = await res.json()
    console.log("Query created:", data)
    return NextResponse.json(data)
  } catch (error) {
    console.error("Query create error:", error)
    return NextResponse.json({ error: "Internal error" }, { status: 500 })
  }
}
