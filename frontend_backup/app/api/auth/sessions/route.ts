import { NextRequest, NextResponse } from "next/server"

export async function GET(request: NextRequest) {
  try {
    const cookie = request.headers.get("cookie") || ""
    const { searchParams } = new URL(request.url)
    const page = searchParams.get("page") || "1"
    const limit = searchParams.get("limit") || "10"
    
    const res = await fetch(`http://localhost:8080/api/auth/sessions?page=${page}&limit=${limit}`, {
      headers: { Cookie: cookie },
    })
    const data = await res.json()
    return NextResponse.json(data, { status: res.status })
  } catch (error) {
    return NextResponse.json({ sessions: [], total: 0 }, { status: 200 })
  }
}
