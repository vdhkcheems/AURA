import { NextResponse } from "next/server";

import { AuraServiceError, chatRequestSchema, streamAnswerQuestion } from "@/lib/aura";

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
    const result = await streamAnswerQuestion(parsed.data);
    const encoder = new TextEncoder();
    const stream = new ReadableStream({
      async start(controller) {
        const send = (event: object) => controller.enqueue(encoder.encode(`${JSON.stringify(event)}\n`));
        try {
          send({ type: "meta", sources: result.sources, paperIds: result.paperIds, model: result.model, warnings: result.warnings });
          if ("fallback" in result) {
            send({ type: "delta", text: result.fallback });
          } else {
            for await (const chunk of result.stream) {
              if (chunk.text) send({ type: "delta", text: chunk.text });
            }
          }
          send({ type: "done" });
        } catch (error) {
          console.error("AURA chat stream failed", error);
          const message = error instanceof AuraServiceError
            ? error.message
            : "The research service stopped while generating an answer. Please try again.";
          send({ type: "error", error: message });
        } finally {
          controller.close();
        }
      },
    });
    return new Response(stream, {
      headers: {
        "content-type": "application/x-ndjson; charset=utf-8",
        "cache-control": "no-cache, no-transform",
        "x-accel-buffering": "no",
      },
    });
  } catch (error) {
    console.error("AURA chat request failed", error);
    const message = error instanceof AuraServiceError
      ? error.message
      : "The research service is temporarily unavailable. Please try again.";
    return NextResponse.json({ error: message }, { status: 502 });
  }
}
