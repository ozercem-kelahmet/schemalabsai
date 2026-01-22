import { NextRequest, NextResponse } from "next/server"

export async function PUT(req: NextRequest) {
  return handleUpdate(req)
}

export async function POST(req: NextRequest) {
  return handleUpdate(req)
}

async function handleUpdate(req: NextRequest) {
  try {
    const body = await req.json()
    const { id, ...updates } = body
    
    if (!id) {
      return NextResponse.json({ error: "ID required" }, { status: 400 })
    }
    
    const apiUrl = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8080"
    const res = await fetch(apiUrl + "/api/queries/update", {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        "Cookie": req.headers.get("cookie") || "",
      },
      body: JSON.stringify({ id, ...updates }),
    })
    
    const data = await res.json()
    return NextResponse.json(data)
  } catch (error) {
    console.error("Update query error:", error)
    return NextResponse.json({ error: "Failed to update" }, { status: 500 })
  }
}
