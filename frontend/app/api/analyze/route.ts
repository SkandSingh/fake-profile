import { NextRequest, NextResponse } from 'next/server'

// Types for the API
interface ProfileInput {
  type: 'url' | 'file'
  url?: string
  fileData?: {
    name: string
    type: string
    size: number
    content?: string // Base64 encoded content for images
  }
  textContent?: string
  profileData?: {
    username?: string
    displayName?: string
    bio?: string
    followerCount?: number
    followingCount?: number
    postCount?: number
    accountAge?: number // in days
    verified?: boolean
    profileImageUrl?: string
  }
}

interface AnalysisResult {
  trustScore: number
  textScore: number
  imageScore: number
  metricsScore: number
  explanation: string[]
  breakdown: {
    textAnalysis: {
      sentiment: 'positive' | 'negative' | 'neutral'
      sentimentScore: number
      toxicity: number
      authenticity: number
      readability: number
      keywords: string[]
      languageDetected: string
      confidence: number
    }
    imageAnalysis: {
      faceDetected: boolean
      imageQuality: number
      manipulation: number
      metadata: {
        originalSource: boolean
        dateConsistency: boolean
        locationConsistency: boolean
      }
      similarImages: number
      confidence: number
    }
    profileMetrics: {
      accountAge: number
      followersToFollowing: number
      engagement: {
        avgLikes: number
        avgComments: number
        avgShares: number
        rate: number
      }
      activityPattern: 'consistent' | 'suspicious' | 'normal'
      verification: {
        email: boolean
        phone: boolean
        identity: boolean
      }
      riskFactors: string[]
    }
  }
  processingTime: number
  analysisId: string
}

// Mock ML model functions (to be replaced with real implementations)
async function analyzeText(textContent: string): Promise<any> {
  // Simulate ML processing time
  await new Promise(resolve => setTimeout(resolve, 500))
  
  // Mock text analysis based on content
  const wordCount = textContent.split(' ').length
  const hasPositiveWords = /happy|great|amazing|love|excellent|wonderful|good/i.test(textContent)
  const hasNegativeWords = /hate|terrible|awful|bad|horrible|worst/i.test(textContent)
  const hasToxicWords = /stupid|idiot|kill|die|hate/i.test(textContent)
  
  let sentiment: 'positive' | 'negative' | 'neutral' = 'neutral'
  let sentimentScore = 50
  
  if (hasPositiveWords && !hasNegativeWords) {
    sentiment = 'positive'
    sentimentScore = Math.random() * 30 + 70 // 70-100
  } else if (hasNegativeWords && !hasPositiveWords) {
    sentiment = 'negative'
    sentimentScore = Math.random() * 30 + 10 // 10-40
  } else {
    sentimentScore = Math.random() * 40 + 30 // 30-70
  }
  
  return {
    sentiment,
    sentimentScore: Math.round(sentimentScore),
    toxicity: hasToxicWords ? Math.random() * 50 + 30 : Math.random() * 20,
    authenticity: wordCount > 50 ? Math.random() * 20 + 80 : Math.random() * 30 + 60,
    readability: Math.random() * 30 + 70,
    keywords: extractKeywords(textContent),
    languageDetected: 'English',
    confidence: Math.random() * 10 + 90
  }
}

async function analyzeImage(fileData: any): Promise<any> {
  // Simulate ML processing time
  await new Promise(resolve => setTimeout(resolve, 800))
  
  // Mock image analysis based on file properties
  const isLargeFile = fileData.size > 1000000 // > 1MB
  const isPNG = fileData.type === 'image/png'
  const isJPG = fileData.type === 'image/jpeg'
  
  return {
    faceDetected: Math.random() > 0.3,
    imageQuality: isLargeFile ? Math.random() * 20 + 80 : Math.random() * 30 + 60,
    manipulation: isPNG ? Math.random() * 15 : Math.random() * 25,
    metadata: {
      originalSource: Math.random() > 0.2,
      dateConsistency: Math.random() > 0.3,
      locationConsistency: Math.random() > 0.4
    },
    similarImages: Math.floor(Math.random() * 10),
    confidence: Math.random() * 15 + 85
  }
}

async function analyzeProfileMetrics(profileData: any): Promise<any> {
  // Simulate ML processing time
  await new Promise(resolve => setTimeout(resolve, 300))
  
  const followerCount = profileData.followerCount || 0
  const followingCount = profileData.followingCount || 0
  const accountAge = profileData.accountAge || 30
  const postCount = profileData.postCount || 0
  
  const followersToFollowing = followingCount > 0 ? followerCount / followingCount : 0
  const postsPerDay = accountAge > 0 ? postCount / accountAge : 0
  
  // Calculate engagement rate (mock)
  const engagementRate = followerCount > 0 
    ? Math.min((postsPerDay * 100) / (followerCount * 0.01), 10)
    : Math.random() * 5
  
  let activityPattern: 'consistent' | 'suspicious' | 'normal' = 'normal'
  if (postsPerDay > 10) activityPattern = 'suspicious'
  else if (postsPerDay > 1 && postsPerDay < 5) activityPattern = 'consistent'
  
  const riskFactors: string[] = []
  if (accountAge < 30) riskFactors.push('New account')
  if (followersToFollowing > 10) riskFactors.push('High followers-to-following ratio')
  if (postsPerDay > 10) riskFactors.push('Excessive posting frequency')
  if (!profileData.verified) riskFactors.push('Unverified account')
  
  return {
    accountAge,
    followersToFollowing: Math.round(followersToFollowing * 100) / 100,
    engagement: {
      avgLikes: Math.floor(followerCount * 0.05),
      avgComments: Math.floor(followerCount * 0.01),
      avgShares: Math.floor(followerCount * 0.005),
      rate: Math.round(engagementRate * 100) / 100
    },
    activityPattern,
    verification: {
      email: Math.random() > 0.3,
      phone: Math.random() > 0.5,
      identity: profileData.verified || Math.random() > 0.8
    },
    riskFactors
  }
}

function extractKeywords(text: string): string[] {
  // Simple keyword extraction (to be replaced with NLP)
  const words = text.toLowerCase()
    .replace(/[^\w\s]/g, '')
    .split(/\s+/)
    .filter(word => word.length > 3)
  
  const stopWords = new Set(['that', 'this', 'with', 'have', 'will', 'been', 'from', 'they', 'know', 'want', 'been', 'good', 'much', 'some', 'time', 'very', 'when', 'come', 'here', 'just', 'like', 'long', 'make', 'many', 'over', 'such', 'take', 'than', 'them', 'well', 'were'])
  
  const filteredWords = words.filter(word => !stopWords.has(word))
  const wordCount = filteredWords.reduce((acc, word) => {
    acc[word] = (acc[word] || 0) + 1
    return acc
  }, {} as Record<string, number>)
  
  return Object.entries(wordCount)
    .sort(([,a], [,b]) => b - a)
    .slice(0, 8)
    .map(([word]) => word)
}

function calculateTrustScore(textScore: number, imageScore: number, metricsScore: number): number {
  // Weighted calculation
  const weights = {
    text: 0.35,
    image: 0.25,
    metrics: 0.40
  }
  
  return Math.round(
    textScore * weights.text +
    imageScore * weights.image +
    metricsScore * weights.metrics
  )
}

function generateExplanations(textScore: number, imageScore: number, metricsScore: number, trustScore: number): string[] {
  const explanations: string[] = []
  
  // Overall assessment
  if (trustScore >= 80) {
    explanations.push("Profile shows strong indicators of authenticity and trustworthiness.")
  } else if (trustScore >= 60) {
    explanations.push("Profile appears moderately trustworthy with some areas requiring attention.")
  } else if (trustScore >= 40) {
    explanations.push("Profile raises several concerns that warrant careful consideration.")
  } else {
    explanations.push("Profile exhibits multiple risk factors suggesting potential fraudulent activity.")
  }
  
  // Text analysis insights
  if (textScore >= 80) {
    explanations.push("Text content demonstrates authentic language patterns and positive sentiment.")
  } else if (textScore < 50) {
    explanations.push("Text analysis reveals concerning patterns in language use or sentiment.")
  }
  
  // Image analysis insights
  if (imageScore >= 80) {
    explanations.push("Profile images show high quality with no signs of manipulation.")
  } else if (imageScore < 50) {
    explanations.push("Image analysis detected potential manipulation or quality issues.")
  }
  
  // Metrics insights
  if (metricsScore >= 80) {
    explanations.push("Account metrics indicate natural, organic growth and engagement patterns.")
  } else if (metricsScore < 50) {
    explanations.push("Account metrics suggest potential artificial inflation or suspicious activity.")
  }
  
  return explanations
}

export async function POST(request: NextRequest) {
  try {
    const startTime = Date.now()
    
    // Parse the request body
    const body: ProfileInput = await request.json()
    
    // Validate input
    if (!body.type || (body.type === 'url' && !body.url) || (body.type === 'file' && !body.fileData)) {
      return NextResponse.json(
        { error: 'Invalid input: missing required fields' },
        { status: 400 }
      )
    }
    
    // Extract content for analysis
    const textContent = body.textContent || body.profileData?.bio || "Sample profile text for analysis"
    const profileData = body.profileData || {
      followerCount: Math.floor(Math.random() * 10000),
      followingCount: Math.floor(Math.random() * 1000),
      postCount: Math.floor(Math.random() * 500),
      accountAge: Math.floor(Math.random() * 1000) + 30,
      verified: Math.random() > 0.7
    }
    
    // Run analysis (in parallel for better performance)
    const [textAnalysis, imageAnalysis, profileMetrics] = await Promise.all([
      analyzeText(textContent),
      body.fileData ? analyzeImage(body.fileData) : Promise.resolve(null),
      analyzeProfileMetrics(profileData)
    ])
    
    // Calculate individual scores
    const textScore = Math.round(
      (textAnalysis.sentimentScore + textAnalysis.authenticity + (100 - textAnalysis.toxicity)) / 3
    )
    
    const imageScore = imageAnalysis ? Math.round(
      (imageAnalysis.imageQuality + (100 - imageAnalysis.manipulation) + 
       (imageAnalysis.metadata.originalSource ? 100 : 0) + 
       (imageAnalysis.metadata.dateConsistency ? 100 : 0)) / 4
    ) : 75 // Default score if no image
    
    const metricsScore = Math.round(
      Math.min(
        (profileMetrics.accountAge / 365) * 20 + // Age factor (0-20)
        Math.min(profileMetrics.followersToFollowing * 10, 30) + // Ratio factor (0-30)
        profileMetrics.engagement.rate * 10 + // Engagement factor (0-50)
        (profileMetrics.verification.email ? 10 : 0) + // Verification bonuses
        (profileMetrics.verification.phone ? 10 : 0) +
        (profileMetrics.verification.identity ? 10 : 0) -
        (profileMetrics.riskFactors.length * 5), // Risk penalty
        100
      )
    )
    
    // Calculate overall trust score
    const trustScore = calculateTrustScore(textScore, imageScore, metricsScore)
    
    // Generate explanations
    const explanation = generateExplanations(textScore, imageScore, metricsScore, trustScore)
    
    // Create analysis result
    const result: AnalysisResult = {
      trustScore,
      textScore,
      imageScore,
      metricsScore,
      explanation,
      breakdown: {
        textAnalysis,
        imageAnalysis: imageAnalysis || {
          faceDetected: false,
          imageQuality: 75,
          manipulation: 10,
          metadata: {
            originalSource: true,
            dateConsistency: true,
            locationConsistency: true
          },
          similarImages: 0,
          confidence: 85
        },
        profileMetrics
      },
      processingTime: Date.now() - startTime,
      analysisId: `analysis_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`
    }
    
    // Add artificial delay to simulate ML processing
    await new Promise(resolve => setTimeout(resolve, 1000))
    
    return NextResponse.json(result)
    
  } catch (error) {
    console.error('Analysis error:', error)
    return NextResponse.json(
      { error: 'Internal server error during analysis' },
      { status: 500 }
    )
  }
}

// Handle GET requests for API documentation
export async function GET() {
  return NextResponse.json({
    name: 'Profile Analysis API',
    version: '1.0.0',
    description: 'API endpoint for analyzing social media profile trustworthiness',
    endpoints: {
      'POST /api/analyze': {
        description: 'Analyze a profile and return trust score with detailed breakdown',
        body: {
          type: 'url | file',
          url: 'string (optional)',
          fileData: 'object (optional)',
          textContent: 'string (optional)',
          profileData: 'object (optional)'
        },
        response: {
          trustScore: 'number (0-100)',
          textScore: 'number (0-100)',
          imageScore: 'number (0-100)',
          metricsScore: 'number (0-100)',
          explanation: 'string[]',
          breakdown: 'object',
          processingTime: 'number (ms)',
          analysisId: 'string'
        }
      }
    },
    example: {
      request: {
        type: 'url',
        url: 'https://twitter.com/example',
        textContent: 'This is a sample bio text',
        profileData: {
          followerCount: 1000,
          followingCount: 500,
          postCount: 100,
          accountAge: 365,
          verified: false
        }
      }
    }
  })
}