/**
 * Thin wrapper around the Gemini API (plain REST, no SDK dependency) used
 * for the qualitative half of profile analysis: reading bio text (and the
 * profile photo, if provided) together and returning a structured,
 * explainable assessment. The quantitative half (follower ratios, username
 * patterns, etc.) is handled separately by the trained classifier in
 * metricsModel.ts - Gemini is not asked to invent a numeric "trust score"
 * on its own.
 */

const GEMINI_MODEL = process.env.GEMINI_MODEL || "gemini-2.5-flash";
const GEMINI_API_URL = `https://generativelanguage.googleapis.com/v1beta/models/${GEMINI_MODEL}:generateContent`;
const REQUEST_TIMEOUT_MS = 15_000;

export interface GeminiProfileInput {
  displayName: string;
  bio: string;
  platform: string;
  imageBase64?: string;
  imageMimeType?: string;
}

export interface GeminiAnalysis {
  textAuthenticityScore: number; // 0-1, higher = more authentic-sounding
  imageAssessment: {
    looksAuthentic: boolean;
    reasoning: string;
  } | null;
  riskFactors: string[];
  explanation: string;
}

const RESPONSE_SCHEMA = {
  type: "OBJECT",
  properties: {
    textAuthenticityScore: { type: "NUMBER" },
    imageLooksAuthentic: { type: "BOOLEAN" },
    imageReasoning: { type: "STRING" },
    riskFactors: { type: "ARRAY", items: { type: "STRING" } },
    explanation: { type: "STRING" },
  },
  required: ["textAuthenticityScore", "riskFactors", "explanation"],
};

function buildPrompt(input: GeminiProfileInput): string {
  return `You are assessing whether a ${input.platform} profile looks authentic based on its bio text${
    input.imageBase64 ? " and profile photo" : ""
  }. Be skeptical of generic engagement-bait language, but do not penalize normal, boring, real-looking bios.

Display name: ${input.displayName || "(none provided)"}
Bio: ${input.bio || "(empty bio)"}

Respond with JSON matching the schema: textAuthenticityScore (0-1, where 1 is clearly authentic), ${
    input.imageBase64 ? "imageLooksAuthentic (boolean), imageReasoning (one sentence), " : ""
  }riskFactors (short strings, empty array if none), and explanation (2-3 sentences a non-technical user can read).`;
}

/** Returns null on any failure (timeout, bad key, malformed response) - callers must fall back gracefully, never fabricate a score in its place. */
export async function analyzeWithGemini(input: GeminiProfileInput): Promise<GeminiAnalysis | null> {
  const apiKey = process.env.GEMINI_API_KEY;
  if (!apiKey) {
    return null;
  }

  const parts: Record<string, unknown>[] = [{ text: buildPrompt(input) }];
  if (input.imageBase64 && input.imageMimeType) {
    parts.push({
      inline_data: {
        mime_type: input.imageMimeType,
        data: input.imageBase64,
      },
    });
  }

  const controller = new AbortController();
  const timeout = setTimeout(() => controller.abort(), REQUEST_TIMEOUT_MS);

  try {
    const response = await fetch(GEMINI_API_URL, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        "x-goog-api-key": apiKey,
      },
      signal: controller.signal,
      body: JSON.stringify({
        contents: [{ role: "user", parts }],
        generationConfig: {
          responseMimeType: "application/json",
          responseSchema: RESPONSE_SCHEMA,
        },
      }),
    });

    if (!response.ok) {
      return null;
    }

    const data = await response.json();
    const text: string | undefined = data?.candidates?.[0]?.content?.parts?.[0]?.text;
    if (!text) return null;

    const parsed = JSON.parse(text);

    const score = Number(parsed.textAuthenticityScore);
    if (!Number.isFinite(score)) return null;

    return {
      textAuthenticityScore: Math.min(1, Math.max(0, score)),
      imageAssessment:
        input.imageBase64 && typeof parsed.imageLooksAuthentic === "boolean"
          ? { looksAuthentic: parsed.imageLooksAuthentic, reasoning: String(parsed.imageReasoning || "") }
          : null,
      riskFactors: Array.isArray(parsed.riskFactors) ? parsed.riskFactors.map(String) : [],
      explanation: String(parsed.explanation || ""),
    };
  } catch {
    return null;
  } finally {
    clearTimeout(timeout);
  }
}
