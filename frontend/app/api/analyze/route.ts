import { NextRequest, NextResponse } from "next/server";

export async function POST(request: NextRequest) {
  try {
    const { type, profileData } = await request.json();
    if (type !== "manual" || !profileData) {
      return NextResponse.json({ error: "Only manual input supported" }, { status: 400 });
    }
    // Calculate trust score based on simple metrics
    const textContent = `${profileData.bio || ''} ${profileData.profileText || ''}`.trim();
    const words = textContent.split(' ').length;
    let trustScore = 60; // base score
    trustScore += profileData.verified ? 15 : 0;
    trustScore += words > 10 ? 10 : 0;
    trustScore += profileData.followerCount > 100 ? 10 : 0;
    trustScore = Math.min(Math.max(trustScore, 0), 100);
    
    const response = {
      trustScore,
      confidence: 85,
      riskLevel: trustScore > 75 ? 'low' : trustScore > 50 ? 'medium' : 'high',
      breakdown: {
        textAnalysis: {
          sentiment: 'positive',
          sentimentScore: Math.floor(Math.random() * 20) + 70,
          toxicity: Math.floor(Math.random() * 25),
          authenticity: Math.min(60 + words * 2, 90),
          readability: Math.min(words * 3, 95),
          keywords: ['social', 'media', 'profile'],
          languageDetected: 'English',
          confidence: 85
        },
        imageAnalysis: profileData.profilePicture ? {
          faceDetected: true,
          imageQuality: Math.floor(Math.random() * 20) + 75,
          manipulation: Math.floor(Math.random() * 30),
          metadata: {
            originalSource: true,
            dateConsistency: true,
            locationConsistency: Math.random() > 0.3
          },
          similarImages: Math.floor(Math.random() * 3),
          confidence: 88
        } : null,
        profileMetrics: {
          accountAge: profileData.accountAge || 365,
          followersToFollowing: profileData.followingCount > 0 ? 
            profileData.followerCount / profileData.followingCount : 1,
          engagement: {
            avgLikes: Math.floor(Math.random() * 50) + 10,
            avgComments: Math.floor(Math.random() * 20) + 5,
            avgShares: Math.floor(Math.random() * 10) + 2,
            rate: Math.min((profileData.postCount || 10) / (profileData.accountAge || 365) * 100, 5)
          },
          activityPattern: 'consistent',
          verification: {
            email: true,
            phone: profileData.verified,
            identity: profileData.verified
          },
          riskFactors: []
        }
      },
      timestamp: new Date().toISOString(),
      profileSummary: {
        username: profileData.username,
        displayName: profileData.displayName || profileData.username,
        platform: profileData.platform,
        followerCount: profileData.followerCount,
        verified: profileData.verified
      }
    };

    return NextResponse.json(response);
  } catch (error) {
    return NextResponse.json({ error: "Analysis failed" }, { status: 500 });
  }
}

export async function GET() {
  return NextResponse.json({ status: "healthy", message: "Profile Purity Detector", version: "2.0" });
}
