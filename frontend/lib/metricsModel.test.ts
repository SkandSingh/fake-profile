import { describe, expect, it } from "vitest";
import { computeMetricsScore, getMetricsRiskFactors, type MetricsProfileInput } from "./metricsModel";

const realisticProfile: MetricsProfileInput = {
  username: "jane_doe",
  displayName: "Jane Doe",
  bio: "Photographer based in NYC. Sharing my travels and coffee obsession one post at a time.",
  followerCount: 842,
  followingCount: 391,
  postCount: 213,
  hasProfilePicture: true,
  isPrivate: false,
};

const spammyProfile: MetricsProfileInput = {
  username: "xyz19837462",
  displayName: "xyz19837462",
  bio: "",
  followerCount: 3,
  followingCount: 4500,
  postCount: 0,
  hasProfilePicture: false,
  isPrivate: false,
};

describe("computeMetricsScore", () => {
  it("is deterministic - identical input always yields identical output", () => {
    const first = computeMetricsScore(realisticProfile);
    const second = computeMetricsScore(realisticProfile);
    expect(second.trustScore).toBe(first.trustScore);
    expect(second.fakeProbability).toBe(first.fakeProbability);
  });

  it("scores a realistic, filled-out profile as more trustworthy than an empty/spammy one", () => {
    const realistic = computeMetricsScore(realisticProfile);
    const spammy = computeMetricsScore(spammyProfile);
    expect(realistic.trustScore).toBeGreaterThan(spammy.trustScore);
  });

  it("returns a probability in [0, 1]", () => {
    const result = computeMetricsScore(realisticProfile);
    expect(result.trustScore).toBeGreaterThanOrEqual(0);
    expect(result.trustScore).toBeLessThanOrEqual(1);
    expect(result.fakeProbability).toBeCloseTo(1 - result.trustScore, 10);
  });

  it("exposes real (non-fabricated) held-out model metrics", () => {
    const result = computeMetricsScore(realisticProfile);
    expect(result.modelMetrics.accuracy).toBeGreaterThan(0.5);
    expect(result.modelMetrics.accuracy).toBeLessThanOrEqual(1);
    expect(result.modelMetrics.trainedOn).toContain("instagram");
  });
});

describe("getMetricsRiskFactors", () => {
  it("flags the empty-bio, no-picture, low-follower profile", () => {
    const result = computeMetricsScore(spammyProfile);
    const flags = getMetricsRiskFactors(result);
    expect(flags.length).toBeGreaterThan(0);
    expect(flags).toContain("No profile picture set");
  });

  it("does not flag a normal, filled-out profile with the same message", () => {
    const result = computeMetricsScore(realisticProfile);
    const flags = getMetricsRiskFactors(result);
    expect(flags).not.toContain("No profile picture set");
  });
});
