import { NextRequest, NextResponse } from "next/server"

export async function POST(request: NextRequest) {
  try {
    const formData = await request.formData()
    const cookie = request.headers.get("cookie") || ""
    
    const res = await fetch("http://localhost:8080/api/auth/upload-avatar", {
      method: "POST",
      headers: { Cookie: cookie },
      body: formData,
    })
    const data = await res.json()
    return NextResponse.json(data, { status: res.status })
  } catch (error) {
    return NextResponse.json({ error: "Failed to upload avatar" }, { status: 500 })
  }
}
