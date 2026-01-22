import { NextRequest, NextResponse } from "next/server"

export async function POST(request: NextRequest) {
  try {
    const cookie = request.headers.get("cookie") || ""
    const res = await fetch("http://localhost:8080/api/auth/change-password-request", {
      method: "POST",
      headers: { Cookie: cookie },
    })
    const data = await res.json()
    return NextResponse.json(data, { status: res.status })
  } catch (error) {
    return NextResponse.json({ error: "Failed to send verification code" }, { status: 500 })
  }
}
