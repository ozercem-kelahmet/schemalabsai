import { NextRequest, NextResponse } from "next/server"

export async function POST(request: NextRequest) {
  try {
    const cookie = request.headers.get("cookie") || ""
    const res = await fetch("http://localhost:8080/api/auth/logout-all", {
      method: "POST",
      headers: { Cookie: cookie },
    })
    const data = await res.json()
    return NextResponse.json(data, { status: res.status })
  } catch (error) {
    return NextResponse.json({ error: "Failed to logout" }, { status: 500 })
  }
}
