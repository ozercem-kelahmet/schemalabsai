import { NextResponse } from "next/server"
import { cookies } from "next/headers"

export async function GET() {
  const session = (await cookies()).get("session")
  if (!session?.value) {
    return NextResponse.json({ error: "Unauthorized" }, { status: 401 })
  }

  return NextResponse.json({
    plan: { name: "Free", price: 0, status: "active", nextBilling: "" },
    paymentMethod: null,
    invoices: []
  })
}
