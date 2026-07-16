import { NextRequest, NextResponse } from "next/server";
import * as cheerio from "cheerio";

/**
 * Best-effort public-profile extraction via Open Graph meta tags.
 *
 * Both Instagram and X/Twitter serve `og:title`/`og:description`/`og:image`
 * meta tags on public profile pages without requiring login. Instagram's
 * og:description has historically embedded "X Followers, Y Following, Z
 * Posts - ..." counts; X/Twitter's og:description is typically just the
 * bio, with no counts. Private profiles, login walls, and platform-side
 * blocking all mean this can legitimately fail - the caller should treat
 * `manualInputRequired: true` (or any missing field) as "ask the user to
 * fill in the rest," never as an error.
 */

type Platform = "instagram" | "twitter" | "unknown";

interface ExtractedProfileData {
  platform: Platform;
  username: string;
  displayName: string;
  bio: string;
  followerCount: number;
  followingCount: number;
  postCount: number;
  profileImageUrl: string;
  extractionMethod: string;
  manualInputRequired: boolean;
  missingFields: string[];
  extractionError?: string;
}

const FETCH_TIMEOUT_MS = 10_000;
const BROWSER_USER_AGENT =
  "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36";

function detectPlatform(url: URL): Platform {
  const host = url.hostname.replace(/^www\./, "");
  if (host === "instagram.com") return "instagram";
  if (host === "twitter.com" || host === "x.com") return "twitter";
  return "unknown";
}

function extractUsername(url: URL): string {
  const segments = url.pathname.split("/").filter(Boolean);
  return segments[0] || "";
}

function parseCount(raw: string): number {
  const cleaned = raw.trim().toUpperCase().replace(/,/g, "");
  const multiplier = cleaned.endsWith("K") ? 1_000 : cleaned.endsWith("M") ? 1_000_000 : 1;
  const numeric = parseFloat(cleaned.replace(/[KM]$/, ""));
  return Number.isFinite(numeric) ? Math.round(numeric * multiplier) : 0;
}

/** Instagram's og:description is typically "12.3K Followers, 456 Following, 78 Posts - ..." */
function parseInstagramCounts(description: string) {
  const followerMatch = description.match(/([\d,.]+[KM]?)\s+Followers?/i);
  const followingMatch = description.match(/([\d,.]+[KM]?)\s+Following/i);
  const postMatch = description.match(/([\d,.]+[KM]?)\s+Posts?/i);
  return {
    followerCount: followerMatch ? parseCount(followerMatch[1]) : null,
    followingCount: followingMatch ? parseCount(followingMatch[1]) : null,
    postCount: postMatch ? parseCount(postMatch[1]) : null,
  };
}

async function fetchHtml(url: string): Promise<string> {
  const controller = new AbortController();
  const timeout = setTimeout(() => controller.abort(), FETCH_TIMEOUT_MS);
  try {
    const response = await fetch(url, {
      headers: {
        "User-Agent": BROWSER_USER_AGENT,
        Accept: "text/html,application/xhtml+xml",
        "Accept-Language": "en-US,en;q=0.9",
      },
      redirect: "follow",
      signal: controller.signal,
      cache: "no-store",
    });
    if (!response.ok) {
      throw new Error(`Profile page responded with ${response.status}`);
    }
    return await response.text();
  } finally {
    clearTimeout(timeout);
  }
}

function fallback(platform: Platform, username: string, error: string): ExtractedProfileData {
  return {
    platform,
    username,
    displayName: "",
    bio: "",
    followerCount: 0,
    followingCount: 0,
    postCount: 0,
    profileImageUrl: "",
    extractionMethod: "none",
    manualInputRequired: true,
    missingFields: ["displayName", "bio", "followerCount", "followingCount", "postCount", "profileImageUrl"],
    extractionError: error,
  };
}

export async function POST(request: NextRequest) {
  let body: { url?: string };
  try {
    body = await request.json();
  } catch {
    return NextResponse.json({ error: "Request body must be JSON" }, { status: 400 });
  }

  if (!body.url || typeof body.url !== "string") {
    return NextResponse.json({ error: "A profile 'url' string is required" }, { status: 400 });
  }

  let parsedUrl: URL;
  try {
    parsedUrl = new URL(body.url);
  } catch {
    return NextResponse.json({ error: "Invalid URL" }, { status: 400 });
  }

  const platform = detectPlatform(parsedUrl);
  const username = extractUsername(parsedUrl);

  if (platform === "unknown") {
    return NextResponse.json(fallback(platform, username, "Only Instagram and Twitter/X URLs are supported"));
  }
  if (!username) {
    return NextResponse.json(fallback(platform, username, "Could not determine a username from the URL"));
  }

  try {
    const html = await fetchHtml(parsedUrl.toString());
    const $ = cheerio.load(html);

    const ogTitle = $('meta[property="og:title"]').attr("content") || "";
    const ogDescription = $('meta[property="og:description"]').attr("content") || "";
    const ogImage = $('meta[property="og:image"]').attr("content") || "";

    if (!ogTitle && !ogDescription) {
      return NextResponse.json(
        fallback(platform, username, "Profile page did not expose the expected metadata (likely a login wall or private account)")
      );
    }

    const displayName = ogTitle.split(/\s*[(•|]/)[0].trim();
    const missingFields: string[] = [];

    let followerCount = 0;
    let followingCount = 0;
    let postCount = 0;
    let bio = ogDescription;

    if (platform === "instagram") {
      const counts = parseInstagramCounts(ogDescription);
      followerCount = counts.followerCount ?? 0;
      followingCount = counts.followingCount ?? 0;
      postCount = counts.postCount ?? 0;
      // Instagram's description is "counts - See Instagram photos ... from Name (@user)",
      // not the actual bio - the real bio isn't reliably available without login.
      bio = "";
      if (counts.followerCount === null) missingFields.push("followerCount");
      if (counts.followingCount === null) missingFields.push("followingCount");
      if (counts.postCount === null) missingFields.push("postCount");
      missingFields.push("bio");
    } else {
      // Twitter/X's og:description is the bio itself; counts aren't exposed here.
      missingFields.push("followerCount", "followingCount", "postCount");
    }

    if (!ogImage) missingFields.push("profileImageUrl");
    if (!displayName) missingFields.push("displayName");

    const result: ExtractedProfileData = {
      platform,
      username,
      displayName,
      bio,
      followerCount,
      followingCount,
      postCount,
      profileImageUrl: ogImage,
      extractionMethod: "open_graph_meta_tags",
      manualInputRequired: missingFields.length > 0,
      missingFields,
    };

    return NextResponse.json(result);
  } catch (error) {
    return NextResponse.json(
      fallback(platform, username, error instanceof Error ? error.message : "Extraction failed")
    );
  }
}

export async function GET() {
  return NextResponse.json({ status: "healthy", message: "Profile extraction endpoint" });
}
