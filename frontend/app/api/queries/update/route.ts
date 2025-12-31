import { NextRequest, NextResponse } from "next/server"
import { cookies } from "next/headers"

const BACKEND_URL = process.env.BACKEND_URL || "http://localhost:8080"

export async function POST(req: NextRequest) {
  try {
    const session = (await cookies()).get("session")
    if (!session?.value) {
      return NextResponse.json({ error: "Unauthorized" }, { status: 401 })
    }

    const body = await req.json()
    console.log("Update query request:", body)

    const res = await fetch(BACKEND_URL + "/api/queries/update", {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        Cookie: "session=" + session.value,
      },
      body: JSON.stringify({
        id: body.id,
        name: body.name,
        isTraining: body.isTraining,
        hasModel: body.hasModel,
        trainingModelId: body.trainingModelId
      }),
    })

    if (!res.ok) {
      const errText = await res.text()
      console.error("Backend update error:", errText)
      return NextResponse.json({ error: "Failed to update query" }, { status: res.status })
    }

    const data = await res.json()
    console.log("Query updated:", data)
    return NextResponse.json(data)
  } catch (error) {
    console.error("Query update error:", error)
    return NextResponse.json({ error: "Internal error" }, { status: 500 })
  }
}
