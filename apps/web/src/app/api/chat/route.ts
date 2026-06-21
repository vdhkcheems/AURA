import { NextResponse } from "next/server";

import { answerQuestion, AuraServiceError, chatRequestSchema } from "@/lib/aura";

export const runtime = "nodejs";
export const dynamic = "force-dynamic";

export async function POST(request: Request) {
  let body: unknown;
  try {
    body = await request.json();
  } catch {
    return NextResponse.json({ error: "Request body must be valid JSON." }, { status: 400 });
  }

  const parsed = chatRequestSchema.safeParse(body);
  if (!parsed.success) {
    return NextResponse.json(
      { error: "Invalid chat request.", details: parsed.error.issues },
      { status: 400 },
    );
  }

  try {
    return NextResponse.json(await answerQuestion(parsed.data));
  } catch (error) {
    console.error("AURA chat request failed", error);
    const message = error instanceof AuraServiceError
      ? error.message
      : "The research service is temporarily unavailable. Please try again.";
    return NextResponse.json({ error: message }, { status: 502 });
  }
}
