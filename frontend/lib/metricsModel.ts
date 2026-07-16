import weights from "./metricsModel.json";

/**
 * Deterministic TypeScript port of the logistic regression trained offline
 * in backend/ml/train_metrics_model.py on a labeled Instagram fake-account
 * dataset. Same math as training (normalize -> dot product -> sigmoid), no
 * network call, no randomness - the same profile always yields the same
 * score.
 */
export interface MetricsProfileInput {
  username: string;
  displayName: string;
  bio: string;
  followerCount: number;
  followingCount: number;
  postCount: number;
  hasProfilePicture: boolean;
  isPrivate: boolean;
  externalUrl?: string;
}

interface FeatureContribution {
  key: string;
  value: number;
  contribution: number;
}

export interface MetricsScoreResult {
  trustScore: number; // 0-1, higher = more likely a real/authentic account
  fakeProbability: number; // 0-1, the model's raw P(fake)
  contributions: FeatureContribution[];
  modelMetrics: {
    accuracy: number;
    precision: number;
    recall: number;
    auc: number;
    trainedOn: string;
  };
}

function digitRatio(value: string): number {
  if (value.length === 0) return 0;
  const digitCount = (value.match(/\d/g) || []).length;
  return digitCount / value.length;
}

function countWords(value: string): number {
  return value.trim().length === 0 ? 0 : value.trim().split(/\s+/).length;
}

const URL_PATTERN = /https?:\/\/\S+|(?:^|\s)www\.\S+/i;

function hasUrl(...texts: (string | undefined)[]): boolean {
  return texts.some((text) => !!text && URL_PATTERN.test(text));
}

/** Builds the 11-feature vector in the exact order the model was trained on. */
function extractFeatureVector(profile: MetricsProfileInput): number[] {
  const nameEqualsUsername =
    profile.displayName.trim().toLowerCase() === profile.username.trim().toLowerCase() ? 1 : 0;

  return [
    profile.hasProfilePicture ? 1 : 0, // hasProfilePic
    digitRatio(profile.username), // usernameDigitRatio
    countWords(profile.displayName), // fullnameWordCount
    digitRatio(profile.displayName), // fullnameDigitRatio
    nameEqualsUsername, // nameEqualsUsername
    profile.bio.length, // descriptionLength
    hasUrl(profile.externalUrl, profile.bio) ? 1 : 0, // hasExternalUrl
    profile.isPrivate ? 1 : 0, // isPrivate
    profile.postCount, // postCount
    profile.followerCount, // followerCount
    profile.followingCount, // followCount
  ];
}

function sigmoid(z: number): number {
  return 1 / (1 + Math.exp(-z));
}

const RISK_FACTOR_MESSAGES: Record<string, string> = {
  hasProfilePic: "No profile picture set",
  usernameDigitRatio: "Username is mostly digits",
  fullnameDigitRatio: "Display name contains an unusual number of digits",
  nameEqualsUsername: "Display name is identical to the username",
  descriptionLength: "Bio is very short or empty",
  hasExternalUrl: "No external link in bio",
  postCount: "Very few posts",
  followerCount: "Very few followers",
};

/** Turns the model's real per-feature contributions into human-readable flags, instead of inventing them. */
export function getMetricsRiskFactors(result: MetricsScoreResult, threshold = 0.25): string[] {
  return result.contributions
    .filter((c) => c.contribution > threshold && RISK_FACTOR_MESSAGES[c.key])
    .map((c) => RISK_FACTOR_MESSAGES[c.key]);
}

export function computeMetricsScore(profile: MetricsProfileInput): MetricsScoreResult {
  const rawFeatures = extractFeatureVector(profile);
  const { means, stds, coefficients, intercept, featureKeys, metrics, trainedOn } = weights;

  let z = intercept;
  const contributions: FeatureContribution[] = [];

  for (let i = 0; i < rawFeatures.length; i++) {
    const normalized = (rawFeatures[i] - means[i]) / stds[i];
    const contribution = normalized * coefficients[i];
    z += contribution;
    contributions.push({ key: featureKeys[i], value: rawFeatures[i], contribution });
  }

  const fakeProbability = sigmoid(z);

  return {
    trustScore: 1 - fakeProbability,
    fakeProbability,
    contributions: contributions.sort((a, b) => Math.abs(b.contribution) - Math.abs(a.contribution)),
    modelMetrics: {
      accuracy: metrics.accuracy,
      precision: metrics.precision,
      recall: metrics.recall,
      auc: metrics.auc,
      trainedOn,
    },
  };
}
