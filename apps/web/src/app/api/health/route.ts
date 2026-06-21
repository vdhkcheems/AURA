import { NextResponse } from "next/server";

import { AuraServiceError, checkHealth } from "@/lib/aura";

export const runtime = "nodejs";
export const dynamic = "force-dynamic";

export async function GET() {
  try {
    const health = await checkHealth();
    return NextResponse.json({ status: "ok", ...health });
  } catch (error) {
    console.error("AURA health check failed", error);
    const message = error instanceof AuraServiceError ? error.message : "Qdrant is unavailable.";
    return NextResponse.json({ status: "error", error: message }, { status: 503 });
  }
}
