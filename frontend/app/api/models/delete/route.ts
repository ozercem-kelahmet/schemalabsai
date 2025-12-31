import { NextResponse } from "next/server"
import { cookies } from "next/headers"

const API_URL = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8080"

export async function DELETE(req: Request) {
  try {
    const cookieStore = await cookies()
    const session = cookieStore.get("session")
    
    if (!session) {
      return NextResponse.json({ error: "Unauthorized" }, { status: 401 })
    }

    const { searchParams } = new URL(req.url)
    const id = searchParams.get("id")

    const res = await fetch(`${API_URL}/api/models/finetuned/delete?id=${id}`, {
      method: "DELETE",
      headers: {
        Cookie: `session=${session.value}`,
      },
    })

    const data = await res.json()
    return NextResponse.json(data, { status: res.status })
  } catch (error) {
    console.error("Model delete error:", error)
    return NextResponse.json({ error: "Failed to delete model" }, { status: 500 })
  }
}
