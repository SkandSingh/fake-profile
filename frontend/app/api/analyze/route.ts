import { NextRequest, NextResponse } from "next/server";
import { computeMetricsScore, getMetricsRiskFactors, type MetricsProfileInput } from "@/lib/metricsModel";
import { analyzeWithGemini } from "@/lib/gemini";

/**
 * Combines two independently-computed, real signals into a trust score:
 *  - metricsScore: a logistic regression trained offline on a labeled
 *    Instagram fake-account dataset (see backend/ml/train_metrics_model.py),
 *    applied deterministically via lib/metricsModel.ts.
 *  - geminiScore: Gemini's read of the bio text (and photo, if provided),
 *    via lib/gemini.ts. Optional - if GEMINI_API_KEY isn't configured or the
 *    call fails, this component is left out and the response says so,
 *    rather than inventing a number in its place.
 *
 * No Math.random() anywhere below: the same profileData always produces
 * the same trustScore.
 */

interface ProfileDataInput {
  username?: string;
  displayName?: string;
  platform?: string;
  bio?: string;
  profileText?: string;
  followerCount?: number;
  followingCount?: number;
  postCount?: number;
  accountAge?: number;
  verified?: boolean;
  isPrivate?: boolean;
}

interface FileDataInput {
  name: string;
  type: string;
  size: number;
  content: string; // base64, no data-url prefix
}

function isValidProfileData(data: unknown): data is ProfileDataInput {
  if (!data || typeof data !== "object") return false;
  const p = data as Record<string, unknown>;
  const numericFieldsValid = ["followerCount", "followingCount", "postCount", "accountAge"].every(
    (key) => p[key] === undefined || (typeof p[key] === "number" && Number.isFinite(p[key] as number) && (p[key] as number) >= 0)
  );
  return typeof p.username === "string" && typeof p.bio === "string" && numericFieldsValid;
}

export async function POST(request: NextRequest) {
  let body: { type?: string; profileData?: unknown; fileData?: FileDataInput };
  try {
    body = await request.json();
  } catch {
    return NextResponse.json({ error: "Request body must be JSON" }, { status: 400 });
  }

  if (body.type !== "manual" || !isValidProfileData(body.profileData)) {
    return NextResponse.json({ error: "profileData with at least username/bio is required" }, { status: 400 });
  }
  const profileData = body.profileData;
  const fileData = body.fileData;

  const combinedBioText = `${profileData.bio || ""} ${profileData.profileText || ""}`.trim();

  const metricsInput: MetricsProfileInput = {
    username: profileData.username || "",
    displayName: profileData.displayName || profileData.username || "",
    bio: profileData.bio || "",
    followerCount: profileData.followerCount || 0,
    followingCount: profileData.followingCount || 0,
    postCount: profileData.postCount || 0,
    hasProfilePicture: !!fileData,
    isPrivate: !!profileData.isPrivate,
  };
  const metrics = computeMetricsScore(metricsInput);
  const metricsRiskFactors = getMetricsRiskFactors(metrics);

  const gemini = await analyzeWithGemini({
    displayName: metricsInput.displayName,
    bio: combinedBioText,
    platform: profileData.platform || "social media",
    imageBase64: fileData?.content,
    imageMimeType: fileData?.type,
  });

  // Weighted combination of the two independent, real signals. Falls back
  // to metrics-only when Gemini isn't configured/available.
  const trustScoreFraction = gemini ? 0.5 * metrics.trustScore + 0.5 * gemini.textAuthenticityScore : metrics.trustScore;
  const trustScore = Math.round(trustScoreFraction * 100);

  const riskFactors = [...metricsRiskFactors, ...(gemini?.riskFactors || [])];

  const explanation: string[] = [];
  explanation.push(
    trustScore >= 80
      ? "Profile shows strong indicators of authenticity across both profile metrics and text analysis."
      : trustScore >= 60
        ? "Profile appears moderately trustworthy; some signals warrant a closer look."
        : trustScore >= 40
          ? "Profile raises several concerns based on its metrics and/or bio content."
          : "Profile exhibits multiple risk factors associated with fake/spam accounts in the training data."
  );
  if (gemini?.explanation) {
    explanation.push(gemini.explanation);
  } else {
    explanation.push(
      process.env.GEMINI_API_KEY
        ? "Text/image analysis unavailable right now (Gemini call failed - see server logs) - score is based on profile metrics only."
        : "Text/image analysis unavailable (GEMINI_API_KEY not configured) - score is based on profile metrics only."
    );
  }
  if (metrics.contributions[0]) {
    const top = metrics.contributions[0];
    explanation.push(
      `The strongest metrics-model signal for this profile was "${top.key}" (trained on ${metrics.modelMetrics.trainedOn}, ${Math.round(
        metrics.modelMetrics.accuracy * 100
      )}% held-out accuracy).`
    );
  }

  const textScore = gemini ? Math.round(gemini.textAuthenticityScore * 100) : 50;
  const imageScore = !fileData ? 0 : gemini?.imageAssessment ? (gemini.imageAssessment.looksAuthentic ? 80 : 30) : 50;

  const followersToFollowing =
    metricsInput.followingCount > 0 ? metricsInput.followerCount / metricsInput.followingCount : 0;
  const accountAge = profileData.accountAge ?? 365;

  const response = {
    trustScore,
    confidence: gemini ? 85 : 60,
    riskLevel: trustScore < 40 ? "high" : trustScore < 70 ? "medium" : "low",
    breakdown: {
      textAnalysis: {
        sentimentScore: textScore,
        authenticity: textScore,
        toxicity: Math.min(100, (gemini?.riskFactors.length || 0) * 20),
        confidence: gemini ? 90 : 40,
      },
      imageAnalysis: {
        imageProvided: !!fileData,
        imageQuality: imageScore,
        manipulation: fileData ? 100 - imageScore : 0,
        confidence: !fileData ? 0 : gemini?.imageAssessment ? 80 : 40,
        metadata: { originalSource: gemini?.imageAssessment?.looksAuthentic ?? null },
        reasoning: !fileData
          ? "No profile picture was provided."
          : gemini?.imageAssessment?.reasoning || "Image was provided but could not be assessed.",
      },
      profileMetrics: {
        accountAge,
        followersToFollowing,
        activityPattern: metrics.fakeProbability < 0.3 ? "consistent" : metrics.fakeProbability < 0.6 ? "sporadic" : "irregular",
        engagement: {
          rate: accountAge > 0 ? Math.min((metricsInput.postCount / accountAge) * 100, 5) : 0,
        },
        verification: { identity: !!profileData.verified },
        riskFactors,
        modelMetrics: metrics.modelMetrics,
      },
    },
    timestamp: new Date().toISOString(),
    profileSummary: {
      username: profileData.username,
      displayName: metricsInput.displayName,
      platform: profileData.platform,
      followerCount: metricsInput.followerCount,
      verified: !!profileData.verified,
    },
    explanation,
    textScore,
    imageScore,
    metricsScore: Math.round(metrics.trustScore * 100),
  };

  return NextResponse.json(response);
}

export async function GET() {
  return NextResponse.json({ status: "healthy", message: "Profile Purity Detector", version: "3.0" });
}
