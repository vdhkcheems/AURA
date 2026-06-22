import "server-only";

import { GoogleGenAI } from "@google/genai";
import { z } from "zod";

const vectorSize = 768;
const defaultCollection = "aura_text_ml_core_v1_gemini_embedding_001_v1";
const answerModel = "gemma-4-31b-it";
const minimumGroundedScore = 0.32;
const outOfScopeMessage = "Hi, I’m AURA. I can help you explore the machine-learning research papers in this library—try asking about a concept, equation, experiment, or one of the supported papers.";
const unsupportedPaperMessage = "I don’t yet have a supported paper that gives me evidence for that machine-learning question. We’re expanding AURA’s paper library; for now, try one of the papers listed in the sidebar.";

const historyMessageSchema = z.object({
  role: z.enum(["user", "assistant"]),
  content: z.string().trim().min(1).max(2_000),
});

export const chatRequestSchema = z.object({
  question: z.string().trim().min(1, "A question is required.").max(2_000),
  paperId: z.string().trim().min(1).max(128).optional(),
  topic: z.string().trim().min(1).max(128).optional(),
  history: z.array(historyMessageSchema).max(12).optional().default([]),
});

export type ChatRequest = z.infer<typeof chatRequestSchema>;

export type RetrievedSource = {
  chunkId: string;
  paperId: string;
  title: string;
  sectionPath: string[];
  text: string;
  sourceFiles: string[];
  score: number;
};

export type ChatResponse = {
  answer: string;
  mode: "paper-grounded";
  sources: RetrievedSource[];
  paperIds: string[];
  model: string;
  warnings: string[];
};

type AuraConfig = {
  geminiApiKey: string;
  qdrantUrl: string;
  qdrantApiKey: string;
  qdrantCollection: string;
};

type QdrantPayload = {
  chunk_id?: unknown;
  paper_id?: unknown;
  title?: unknown;
  section_path?: unknown;
  text?: unknown;
  source_files?: unknown;
};

export class AuraServiceError extends Error {}

export async function streamAnswerQuestion(request: ChatRequest) {
  if (isClearlyOutsideResearchScope(request)) {
    return fallbackResponse(outOfScopeMessage, "This question is outside AURA's machine-learning paper library.");
  }

  const config = getConfig();
  const gemini = new GoogleGenAI({ apiKey: config.geminiApiKey });

  const embeddingResponse = await gemini.models.embedContent({
    model: "gemini-embedding-001",
    contents: request.question,
    config: {
      taskType: "QUESTION_ANSWERING",
      outputDimensionality: vectorSize,
    },
  });
  const vector = embeddingResponse.embeddings?.[0]?.values;
  if (!vector || vector.length !== vectorSize) {
    throw new AuraServiceError("Gemini did not return a valid query embedding.");
  }

  const sources = await retrieveSources(config, vector, request);
  const paperIds = [...new Set(sources.map((source) => source.paperId))];
  if (sources.length === 0 || sources[0].score < minimumGroundedScore) {
    return fallbackResponse(unsupportedPaperMessage, "No supported paper provided sufficiently relevant evidence.");
  }

  return {
    sources,
    paperIds,
    model: answerModel,
    warnings: [],
    stream: await gemini.models.generateContentStream({
      model: answerModel,
      config: generationConfig,
      contents: buildGenerationPrompt(request.question, sources, request.history),
    }),
  };
}

export async function answerQuestion(request: ChatRequest): Promise<ChatResponse> {
  const config = getConfig();
  const gemini = new GoogleGenAI({ apiKey: config.geminiApiKey });

  const embeddingResponse = await gemini.models.embedContent({
    model: "gemini-embedding-001",
    contents: request.question,
    config: {
      taskType: "QUESTION_ANSWERING",
      outputDimensionality: vectorSize,
    },
  });
  const vector = embeddingResponse.embeddings?.[0]?.values;
  if (!vector || vector.length !== vectorSize) {
    throw new AuraServiceError("Gemini did not return a valid query embedding.");
  }

  const sources = await retrieveSources(config, vector, request);
  if (sources.length === 0) {
    return {
      answer: "I could not find relevant evidence in the indexed research papers for that question.",
      mode: "paper-grounded",
      sources: [],
      paperIds: [],
      model: answerModel,
      warnings: ["No retrieval results met the query constraints."],
    };
  }

  const generation = await gemini.models.generateContent({
    model: answerModel,
    config: generationConfig,
    contents: buildGenerationPrompt(request.question, sources, request.history),
  });
  const answer = generation.text?.trim();
  if (!answer) {
    throw new AuraServiceError("Gemini did not return an answer.");
  }

  return {
    answer,
    mode: "paper-grounded",
    sources,
    paperIds: [...new Set(sources.map((source) => source.paperId))],
    model: answerModel,
    warnings: [],
  };
}

const generationConfig = {
  temperature: 0.2,
  maxOutputTokens: 5_012,
  systemInstruction: [
    "You are AURA, a research-paper question-answering assistant.",
    "Answer only from the supplied paper evidence.",
    `If the evidence is insufficient for a machine-learning question, respond exactly: "${unsupportedPaperMessage}"`,
    "Do not use outside knowledge to fill gaps.",
    "Use clear technical language and cite sources as [1], [2], and so on, matching the evidence order.",
  ].join(" "),
};

function fallbackResponse(answer: string, warning: string) {
  return {
    sources: [] as RetrievedSource[],
    paperIds: [] as string[],
    model: answerModel,
    warnings: [warning],
    fallback: answer,
  };
}

function isClearlyOutsideResearchScope(request: ChatRequest): boolean {
  const hasGroundedConversation = request.history.some(
    (message) => message.role === "assistant"
      && message.content !== outOfScopeMessage
      && message.content !== unsupportedPaperMessage,
  );
  if (request.paperId || hasGroundedConversation) return false;
  const question = request.question.toLowerCase();
  const researchTerms = /\b(machine learning|\bml\b|deep learning|neural|transformer|attention|bert|gpt|language model|diffusion|computer vision|vision-language|clip|resnet|batch norm|dropout|retrieval|rag\b|dense passage|embedding|vector database|gradient|backprop|training|dataset|benchmark|experiment|paper|research|equation|latex|model)\b/;
  return !researchTerms.test(question);
}

export function getConfig(): AuraConfig {
  const geminiApiKey = process.env.GEMINI_API_KEY;
  const qdrantUrl = process.env.QDRANT_URL;
  const qdrantApiKey = process.env.QDRANT_API_KEY;
  if (!geminiApiKey || !qdrantUrl || !qdrantApiKey) {
    throw new AuraServiceError(
      "AURA is missing GEMINI_API_KEY, QDRANT_URL, or QDRANT_API_KEY server configuration.",
    );
  }
  return {
    geminiApiKey,
    qdrantUrl,
    qdrantApiKey,
    qdrantCollection: process.env.QDRANT_COLLECTION || defaultCollection,
  };
}

export async function checkHealth(): Promise<{ collection: string; pointCount: number }> {
  const config = getConfig();
  const count = await qdrantRequest<{ count: number }>(config, `/collections/${config.qdrantCollection}/points/count`, {
    method: "POST",
    body: JSON.stringify({ exact: true }),
  });
  return { collection: config.qdrantCollection, pointCount: count.count };
}

async function retrieveSources(
  config: AuraConfig,
  vector: number[],
  request: ChatRequest,
): Promise<RetrievedSource[]> {
  const must = [];
  if (request.paperId) {
    must.push({ key: "paper_id", match: { value: request.paperId } });
  }
  if (request.topic) {
    must.push({ key: "topics", match: { any: [request.topic] } });
  }
  const results = await qdrantRequest<{ points: Array<{ score: number; payload?: QdrantPayload }> }>(
    config,
    `/collections/${config.qdrantCollection}/points/query`,
    {
      method: "POST",
      body: JSON.stringify({
        query: vector,
        limit: 5,
        filter: must.length ? { must } : undefined,
        with_payload: true,
        with_vector: false,
      }),
    },
  );
  return results.points.flatMap((result) => toSource(result.score, result.payload ?? {}));
}

async function qdrantRequest<T>(config: AuraConfig, path: string, init: RequestInit): Promise<T> {
  const response = await fetch(`${config.qdrantUrl.replace(/\/$/, "")}${path}`, {
    ...init,
    cache: "no-store",
    headers: {
      "api-key": config.qdrantApiKey,
      "content-type": "application/json",
      ...init.headers,
    },
  });
  if (!response.ok) {
    throw new AuraServiceError(`Qdrant request failed with status ${response.status}.`);
  }
  const envelope = (await response.json()) as { result?: T };
  if (!envelope.result) {
    throw new AuraServiceError("Qdrant returned an unexpected response.");
  }
  return envelope.result;
}

function toSource(score: number, payload: QdrantPayload): RetrievedSource[] {
  if (
    typeof payload.chunk_id !== "string" ||
    typeof payload.paper_id !== "string" ||
    typeof payload.title !== "string" ||
    typeof payload.text !== "string" ||
    !isStringArray(payload.section_path) ||
    !isStringArray(payload.source_files)
  ) {
    return [];
  }
  return [{
    chunkId: payload.chunk_id,
    paperId: payload.paper_id,
    title: payload.title,
    sectionPath: payload.section_path,
    text: payload.text,
    sourceFiles: payload.source_files,
    score,
  }];
}

function isStringArray(value: unknown): value is string[] {
  return Array.isArray(value) && value.every((item) => typeof item === "string");
}

function buildGenerationPrompt(question: string, sources: RetrievedSource[], history: ChatRequest["history"]): string {
  const priorConversation = history.length
    ? history.map((message) => `${message.role === "user" ? "User" : "AURA"}: ${message.content}`).join("\n")
    : "None";
  const evidence = sources
    .map(
      (source, index) =>
        `[${index + 1}] ${source.title}\nSection: ${source.sectionPath.join(" > ")}\n${source.text}`,
    )
    .join("\n\n---\n\n");
  return `Previous conversation:\n${priorConversation}\n\nCurrent question: ${question}\n\nPaper evidence:\n${evidence}`;
}
