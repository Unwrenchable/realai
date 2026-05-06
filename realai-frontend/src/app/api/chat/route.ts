import { NextRequest, NextResponse } from "next/server";
import type { ChatRequest } from "@/lib/realai";

const BACKEND =
  process.env.NEXT_PUBLIC_REALAI_API_BASE ?? "https://realai-qz3b.onrender.com";

export async function POST(req: NextRequest) {
  try {
    const body: ChatRequest = await req.json();
    const { messages, settings } = body;

    // Build the message array, optionally prepending the system prompt.
    const allMessages =
      settings.systemPrompt?.trim()
        ? [
            { role: "system", content: settings.systemPrompt.trim() },
            ...messages,
          ]
        : messages;

    const headers: Record<string, string> = {
      "Content-Type": "application/json",
    };

    // Prefer the server-side env key; fall back to the user-supplied one.
    const apiKey = process.env.REALAI_API_KEY || settings.apiKey;
    if (apiKey) {
      headers["Authorization"] = `Bearer ${apiKey}`;
    }

    const backendRes = await fetch(`${BACKEND}/v1/chat/completions`, {
      method: "POST",
      headers,
      body: JSON.stringify({
        model: settings.model,
        messages: allMessages,
        temperature: settings.temperature,
        max_tokens: settings.maxTokens,
      }),
    });

    const data = await backendRes.json();

    if (!backendRes.ok) {
      return NextResponse.json(
        { error: data?.error?.message ?? "Backend error" },
        { status: backendRes.status }
      );
    }

    return NextResponse.json(data);
  } catch (err) {
    const message = err instanceof Error ? err.message : "Internal server error";
    return NextResponse.json({ error: message }, { status: 500 });
  }
}
