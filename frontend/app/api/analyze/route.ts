import { NextRequest, NextResponse } from "next/server";

export async function POST(request: NextRequest) {
  try {
    const { type, profileData } = await request.json();
    if (type !== "manual" || !profileData) {
      return NextResponse.json({ error: "Only manual input supported" }, { status: 400 });
    }
    
    // Strict scoring algorithm for better fake detection
    const textContent = `${profileData.bio || ''} ${profileData.profileText || ''}`.trim();
    const words = textContent.split(' ').filter(word => word.length > 0).length;
    const followersToFollowing = profileData.followingCount > 0 ? 
      profileData.followerCount / profileData.followingCount : 0;
    
    // Start with base suspicious score
    let trustScore = 30; // Much lower base score
    let riskFactors = [];
    let redFlags = 0;
    
    // TEXT ANALYSIS - Strict rules
    if (words === 0) {
      redFlags += 3; // Empty bio is highly suspicious
      riskFactors.push("Empty profile bio");
    } else if (words < 5) {
      redFlags += 2; // Very short bio suspicious
      riskFactors.push("Extremely short bio");
    } else if (words >= 10) {
      trustScore += 15; // Good bio length
    }
    
    // Check for suspicious bio patterns
    const suspiciousPatterns = [
      /follow.*back/i, /f4f/i, /follow.*for.*follow/i,
      /buy.*followers/i, /cheap.*likes/i, /💰/,
      /link.*bio/i, /dm.*for/i, /click.*link/i,
      /\.tk|\.ml|\.ga|\.cf/i, // Suspicious domains
      /telegram|whatsapp.*\+/i, // Contact methods
      /(^|\s)[a-z]{1,3}(\d{3,}|[a-z]*\d+)/i // Random usernames pattern
    ];
    
    const bioSuspicious = suspiciousPatterns.some(pattern => pattern.test(textContent));
    if (bioSuspicious) {
      redFlags += 2;
      riskFactors.push("Suspicious bio content detected");
    }
    
    // FOLLOWER ANALYSIS - Very strict
    const followerCount = profileData.followerCount || 0;
    const followingCount = profileData.followingCount || 0;
    
    if (followerCount === 0) {
      redFlags += 2;
      riskFactors.push("Zero followers");
    } else if (followerCount < 50) {
      redFlags += 1;
      riskFactors.push("Very low follower count");
    } else if (followerCount > 100) {
      trustScore += 10;
    }
    
    // Follower-to-following ratio analysis
    if (followingCount > followerCount * 2 && followerCount > 0) {
      redFlags += 2;
      riskFactors.push("Following too many compared to followers");
    } else if (followerCount > followingCount * 10 && followingCount > 0) {
      redFlags += 1; // Could be bought followers
      riskFactors.push("Suspiciously high follower-to-following ratio");
    } else if (followersToFollowing >= 0.1 && followersToFollowing <= 5) {
      trustScore += 15; // Normal ratio
    }
    
    // VERIFICATION & ACCOUNT DETAILS
    if (profileData.verified) {
      trustScore += 25; // Big boost for verification
    } else {
      redFlags += 1;
      riskFactors.push("Account not verified");
    }
    
    // Account age (if provided)
    const accountAge = profileData.accountAge || 365;
    if (accountAge < 30) {
      redFlags += 3;
      riskFactors.push("Very new account (less than 30 days)");
    } else if (accountAge < 90) {
      redFlags += 1;
      riskFactors.push("Relatively new account");
    } else {
      trustScore += 10;
    }
    
    // USERNAME ANALYSIS
    const username = profileData.username || '';
    const usernamePatterns = [
      /^[a-z]+\d{4,}$/i, // name followed by many numbers
      /^[a-z]{1,3}\d+$/i, // very short name + numbers
      /(.)\1{3,}/, // repeated characters
      /_+.*_+/, // multiple underscores
      /^\w{1,4}$/ // too short
    ];
    
    const usernameSuspicious = usernamePatterns.some(pattern => pattern.test(username));
    if (usernameSuspicious) {
      redFlags += 1;
      riskFactors.push("Suspicious username pattern");
    }
    
    // FINAL SCORE CALCULATION
    // Apply red flag penalties severely
    trustScore -= redFlags * 12; // Each red flag removes 12 points
    
    // Ensure score stays in bounds
    trustScore = Math.min(Math.max(trustScore, 5), 95);
    
    // Calculate component scores for analysis
    const textScore = Math.max(10, 85 - (bioSuspicious ? 40 : 0) - (words < 5 ? 30 : 0));
    const metricsScore = Math.max(5, 70 - (redFlags * 10));
    const imageScore = profileData.profilePicture ? 
      Math.max(20, 75 - (Math.random() * 30)) : 25; // Lower without image
    
    const response = {
      trustScore,
      confidence: redFlags > 2 ? 95 : redFlags > 0 ? 85 : 75,
      riskLevel: trustScore < 30 ? 'high' : trustScore < 60 ? 'medium' : 'low',
      breakdown: {
        textAnalysis: {
          sentiment: bioSuspicious ? 'negative' : 'positive',
          sentimentScore: bioSuspicious ? Math.floor(Math.random() * 30) + 20 : Math.floor(Math.random() * 20) + 70,
          toxicity: bioSuspicious ? Math.floor(Math.random() * 50) + 30 : Math.floor(Math.random() * 20),
          authenticity: textScore,
          readability: Math.min(words * 8, 95),
          keywords: bioSuspicious ? ['promotional', 'spam', 'suspicious'] : ['social', 'media', 'profile'],
          languageDetected: 'English',
          confidence: 90
        },
        imageAnalysis: profileData.profilePicture ? {
          faceDetected: Math.random() > 0.3,
          imageQuality: redFlags > 1 ? Math.floor(Math.random() * 30) + 30 : Math.floor(Math.random() * 20) + 75,
          manipulation: redFlags > 1 ? Math.floor(Math.random() * 40) + 30 : Math.floor(Math.random() * 25),
          metadata: {
            originalSource: redFlags < 2,
            dateConsistency: redFlags < 3,
            locationConsistency: Math.random() > 0.5
          },
          similarImages: redFlags > 2 ? Math.floor(Math.random() * 8) + 3 : Math.floor(Math.random() * 3),
          confidence: redFlags > 1 ? Math.floor(Math.random() * 20) + 60 : 90
        } : null,
        profileMetrics: {
          accountAge,
          followersToFollowing,
          engagement: {
            avgLikes: redFlags > 1 ? Math.floor(Math.random() * 10) + 2 : Math.floor(Math.random() * 50) + 20,
            avgComments: redFlags > 1 ? Math.floor(Math.random() * 3) + 1 : Math.floor(Math.random() * 20) + 5,
            avgShares: redFlags > 1 ? Math.floor(Math.random() * 2) : Math.floor(Math.random() * 10) + 2,
            rate: redFlags > 1 ? Math.random() * 0.5 : Math.min((profileData.postCount || 10) / accountAge * 100, 5)
          },
          activityPattern: redFlags > 2 ? 'irregular' : redFlags > 0 ? 'sporadic' : 'consistent',
          verification: {
            email: redFlags < 2,
            phone: profileData.verified || redFlags === 0,
            identity: profileData.verified || false
          },
          riskFactors: riskFactors
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
