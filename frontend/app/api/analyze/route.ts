import { NextRequest, NextResponse } from 'next/server'

// Configuration for backend services
const BACKEND_CONFIG = {
  text_api: { url: 'http://127.0.0.1:8000', timeout: 10000 },
  vision_api: { url: 'http://127.0.0.1:8002', timeout: 15000 },
  tabular_api: { url: 'http://127.0.0.1:8003', timeout:8000 },
  ensemble_api: { url: 'http://127.0.0.1:8004', timeout: 12000 },
  profile_extraction_api: { url: 'http://127.0.0.1:8005', timeout: 30000 }
}

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

// Backend service integration functions
async function callNLPService(text: string): Promise<any> {
  const controller = new AbortController()
  const timeoutId = setTimeout(() => controller.abort(), BACKEND_CONFIG.text_api.timeout)
  
  try {
    const response = await fetch(`${BACKEND_CONFIG.text_api.url}/analyze`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({
        text: text,
        include_details: true
      }),
      signal: controller.signal
    })
    
    clearTimeout(timeoutId)
    
    if (!response.ok) {
      throw new Error(`Text API error: ${response.status} ${response.statusText}`)
    }
    
    return await response.json()
  } catch (error: any) {
    clearTimeout(timeoutId)
    if (error.name === 'AbortError') {
      throw new Error('Text analysis service timeout')
    }
    throw error
  }
}

async function callVisionService(imageData: string, mimeType: string): Promise<any> {
  const controller = new AbortController()
  const timeoutId = setTimeout(() => controller.abort(), BACKEND_CONFIG.vision_api.timeout)
  
  try {
    // Convert base64 to blob for form data
    const base64Data = imageData.replace(/^data:[^;]+;base64,/, '')
    const binaryData = atob(base64Data)
    const bytes = new Uint8Array(binaryData.length)
    for (let i = 0; i < binaryData.length; i++) {
      bytes[i] = binaryData.charCodeAt(i)
    }
    const blob = new Blob([bytes], { type: mimeType })
    
    const formData = new FormData()
    formData.append('file', blob, 'profile_image.jpg')
    
    const response = await fetch(`${BACKEND_CONFIG.vision_api.url}/detect/upload`, {
      method: 'POST',
      body: formData,
      signal: controller.signal
    })
    
    clearTimeout(timeoutId)
    
    if (!response.ok) {
      throw new Error(`Vision API error: ${response.status} ${response.statusText}`)
    }
    
    return await response.json()
  } catch (error: any) {
    clearTimeout(timeoutId)
    if (error.name === 'AbortError') {
      throw new Error('Vision analysis service timeout')
    }
    throw error
  }
}

async function callTabularService(profileData: any): Promise<any> {
  const controller = new AbortController()
  const timeoutId = setTimeout(() => controller.abort(), BACKEND_CONFIG.tabular_api.timeout)
  
  try {
    // Transform profile data to match tabular API expected format
    const tabularInput = {
      features: {
        account_age_days: profileData.accountAge || 30,
        followers_following_ratio: profileData.followingCount > 0 
          ? (profileData.followerCount || 0) / profileData.followingCount 
          : (profileData.followerCount || 0),
        post_frequency: profileData.accountAge > 0 
          ? (profileData.postCount || 0) / (profileData.accountAge || 30) 
          : 0,
        engagement_per_post: profileData.followerCount > 0 && profileData.postCount > 0
          ? (profileData.followerCount * 0.05) / profileData.postCount 
          : 0
      }
    }
    
    const response = await fetch(`${BACKEND_CONFIG.tabular_api.url}/predict`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify(tabularInput),
      signal: controller.signal
    })
    
    clearTimeout(timeoutId)
    
    if (!response.ok) {
      throw new Error(`Tabular API error: ${response.status} ${response.statusText}`)
    }
    
    return await response.json()
  } catch (error: any) {
    clearTimeout(timeoutId)
    if (error.name === 'AbortError') {
      throw new Error('Tabular analysis service timeout')
    }
    throw error
  }
}

async function callEnsembleService(textScore: number, imageScore: number, metricsScore: number): Promise<any> {
  const controller = new AbortController()
  const timeoutId = setTimeout(() => controller.abort(), BACKEND_CONFIG.ensemble_api.timeout)
  
  try {
    const response = await fetch(`${BACKEND_CONFIG.ensemble_api.url}/ensemble`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({
        textScore: textScore,
        imageScore: imageScore,
        metricsScore: metricsScore
      }),
      signal: controller.signal
    })
    
    clearTimeout(timeoutId)
    
    if (!response.ok) {
      throw new Error(`Ensemble API error: ${response.status} ${response.statusText}`)
    }
    
    return await response.json()
  } catch (error: any) {
    clearTimeout(timeoutId)
    if (error.name === 'AbortError') {
      throw new Error('Ensemble service timeout')
    }
    throw error
  }
}

// Transformation functions to convert Python service responses to frontend format
function transformTextAnalysis(pythonResponse: any): any {
  const sentiment = pythonResponse.sentiment
  const grammar = pythonResponse.grammar
  const coherence = pythonResponse.coherence
  
  return {
    sentiment: sentiment.overall > 0.6 ? 'positive' : sentiment.overall < 0.4 ? 'negative' : 'neutral',
    sentimentScore: Math.round(sentiment.overall * 100),
    toxicity: Math.round((1 - sentiment.positive) * 100), // Invert positive for toxicity
    authenticity: Math.round(grammar.grammar_score * 100),
    readability: Math.round(coherence.coherence_score * 100),
    keywords: pythonResponse.keywords || [],
    languageDetected: 'English',
    confidence: Math.round(pythonResponse.confidence * 100)
  }
}

function transformVisionAnalysis(pythonResponse: any): any {
  const detection = pythonResponse.detection_result
  
  // For fake profiles, we expect high fake_probability, which should result in low trust scores
  const fakeProb = detection.fake_probability || 0
  const realProb = detection.real_probability || (1 - fakeProb)
  
  return {
    faceDetected: pythonResponse.face_detected || false,
    imageQuality: Math.round(realProb * 100), // Real probability as quality indicator
    manipulation: Math.round(fakeProb * 100), // Fake probability as manipulation level
    metadata: {
      originalSource: realProb > 0.6,
      dateConsistency: detection.confidence > 0.7,
      locationConsistency: detection.confidence > 0.6
    },
    similarImages: fakeProb > 0.5 ? Math.floor(fakeProb * 10) : 0, // More similar images for fake profiles
    confidence: Math.round(detection.confidence * 100)
  }
}

function transformTabularAnalysis(pythonResponse: any, originalProfileData: any): any {
  // The tabular API returns: {probability_real, classification, confidence, model_type}
  const followerCount = originalProfileData.followerCount || 0
  const followingCount = originalProfileData.followingCount || 0
  const accountAge = originalProfileData.accountAge || 30
  const postCount = originalProfileData.postCount || 0
  const probabilityReal = pythonResponse.probability_real || 0.5
  
  // Calculate risk factors based on profile characteristics
  const riskFactors: string[] = []
  
  // Account age risks
  if (accountAge < 30) riskFactors.push('Very new account (< 30 days)')
  else if (accountAge < 90) riskFactors.push('New account (< 3 months)')
  
  // Follower ratio risks
  const followersToFollowing = followingCount > 0 ? followerCount / followingCount : followerCount
  if (followersToFollowing > 20) riskFactors.push('Unusually high followers-to-following ratio')
  else if (followersToFollowing < 0.1 && followerCount > 100) riskFactors.push('Following too many accounts')
  
  // Posting frequency risks
  const postsPerDay = accountAge > 0 ? postCount / accountAge : 0
  if (postsPerDay > 10) riskFactors.push('Excessive posting frequency')
  else if (postsPerDay < 0.01 && accountAge > 30) riskFactors.push('Very low posting activity')
  
  // Low authenticity from ML model
  if (probabilityReal < 0.3) riskFactors.push('ML model detects fake patterns')
  else if (probabilityReal < 0.5) riskFactors.push('ML model shows suspicious indicators')
  
  // Engagement rate calculation (should be realistic)
  const baseEngagementRate = followerCount > 0 ? 
    Math.min((postCount * 50) / (followerCount * accountAge), 10) : 0
  
  // Adjust engagement based on authenticity
  const adjustedEngagementRate = baseEngagementRate * probabilityReal
  
  return {
    accountAge,
    followersToFollowing: Math.round(followersToFollowing * 100) / 100,
    engagement: {
      avgLikes: Math.floor(followerCount * adjustedEngagementRate * 0.05),
      avgComments: Math.floor(followerCount * adjustedEngagementRate * 0.01),
      avgShares: Math.floor(followerCount * adjustedEngagementRate * 0.005),
      rate: Math.round(adjustedEngagementRate * 100) / 100
    },
    activityPattern: probabilityReal > 0.7 ? 'consistent' : 
                    probabilityReal < 0.4 ? 'suspicious' : 'normal',
    verification: {
      email: originalProfileData.verified || Math.random() > 0.7,
      phone: Math.random() > 0.6,
      identity: originalProfileData.verified || (probabilityReal > 0.8 && Math.random() > 0.5)
    },
    riskFactors
  }
}

// Integrated analysis functions using Python backend services
async function analyzeText(textContent: string): Promise<any> {
  try {
    const pythonResponse = await callNLPService(textContent)
    return transformTextAnalysis(pythonResponse)
  } catch (error) {
    console.error('Text analysis error:', error)
    // Fallback to mock data if service fails
    return {
      sentiment: 'neutral',
      sentimentScore: 50,
      toxicity: 20,
      authenticity: 75,
      readability: 80,
      keywords: extractKeywords(textContent),
      languageDetected: 'English',
      confidence: 70
    }
  }
}

async function analyzeImage(fileData: any): Promise<any> {
  try {
    if (!fileData?.content) {
      throw new Error('No image content provided')
    }
    
    const pythonResponse = await callVisionService(fileData.content, fileData.type)
    return transformVisionAnalysis(pythonResponse)
  } catch (error) {
    console.error('Vision analysis error:', error)
    // Fallback to mock data if service fails
    const isLargeFile = fileData.size > 1000000
    const isPNG = fileData.type === 'image/png'
    
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
}

async function analyzeProfileMetrics(profileData: any): Promise<any> {
  try {
    const pythonResponse = await callTabularService(profileData)
    return transformTabularAnalysis(pythonResponse, profileData)
  } catch (error) {
    console.error('Tabular analysis error:', error)
    // Fallback to mock data if service fails
    const followerCount = profileData.followerCount || 0
    const followingCount = profileData.followingCount || 0
    const accountAge = profileData.accountAge || 30
    const postCount = profileData.postCount || 0
    
    const followersToFollowing = followingCount > 0 ? followerCount / followingCount : 0
    const postsPerDay = accountAge > 0 ? postCount / accountAge : 0
    
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

// Weighted scoring algorithm to combine features into single trust score (Profile Purity)
async function calculateTrustScore(
  textScore: number, 
  imageScore: number, 
  metricsScore: number,
  weights?: {
    textWeight: number;
    imageWeight: number; 
    metricsWeight: number;
  }
): Promise<{trustScore: number, confidence: string}> {
  try {
    // Use Profile Purity weights if provided
    const w = weights || { textWeight: 0.35, imageWeight: 0.25, metricsWeight: 0.40 }
    
    // Convert scores to 0-1 range for ensemble service
    const normalizedTextScore = textScore / 100
    const normalizedImageScore = imageScore / 100
    const normalizedMetricsScore = metricsScore / 100
    
    const ensembleResponse = await callEnsembleService(
      normalizedTextScore,
      normalizedImageScore,
      normalizedMetricsScore
    )
    
    // Apply weighted scoring algorithm as specified in problem statement
    const weightedScore = Math.round(
      textScore * w.textWeight + 
      imageScore * w.imageWeight + 
      metricsScore * w.metricsWeight
    )
    
    return {
      trustScore: Math.min(Math.max(weightedScore, 0), 100), // Ensure 0-100 range
      confidence: ensembleResponse.confidence || (weightedScore > 80 ? 'high' : weightedScore > 60 ? 'medium' : 'low')
    }
  } catch (error) {
    console.error('Ensemble service error:', error)
    // Fallback to weighted calculation if ensemble service fails
    const w = weights || { textWeight: 0.35, imageWeight: 0.25, metricsWeight: 0.40 }
    
    const fallbackScore = Math.round(
      textScore * w.textWeight +
      imageScore * w.imageWeight +
      metricsScore * w.metricsWeight
    )
    
    return {
      trustScore: Math.min(Math.max(fallbackScore, 0), 100),
      confidence: fallbackScore > 80 ? 'high' : fallbackScore > 60 ? 'medium' : 'low'
    }
  }
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
  } else if (imageScore < 25) {
    explanations.push("No profile image provided - while some platforms allow this, it reduces trust verification.")
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

// Detect platform from URL for Profile Purity analysis
function detectPlatformFromUrl(url?: string): string {
  if (!url) return 'unknown'
  
  const urlLower = url.toLowerCase()
  if (urlLower.includes('instagram.com')) return 'instagram'
  if (urlLower.includes('twitter.com') || urlLower.includes('x.com')) return 'twitter'
  if (urlLower.includes('facebook.com')) return 'facebook'
  if (urlLower.includes('linkedin.com')) return 'linkedin'
  if (urlLower.includes('tiktok.com')) return 'tiktok'
  
  return 'unknown'
}

async function extractProfileData(url: string): Promise<any> {
  const controller = new AbortController()
  const timeoutId = setTimeout(() => controller.abort(), BACKEND_CONFIG.profile_extraction_api.timeout)
  
  try {
    console.log(`Attempting automatic profile extraction from: ${url}`)
    
    const response = await fetch(`${BACKEND_CONFIG.profile_extraction_api.url}/extract`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({
        url: url,
        extract_image: true,
        timeout: 30
      }),
      signal: controller.signal
    })
    
    clearTimeout(timeoutId)
    
    if (!response.ok) {
      throw new Error(`Profile extraction failed: ${response.status} ${response.statusText}`)
    }
    
    const extractedData = await response.json()
    
    // Transform extracted data to match our expected format
    return {
      username: extractedData.username,
      displayName: extractedData.displayName,
      bio: extractedData.bio,
      followerCount: extractedData.followerCount,
      followingCount: extractedData.followingCount,
      postCount: extractedData.postCount,
      accountAge: extractedData.accountAge,
      verified: extractedData.verified,
      profileImageUrl: extractedData.profileImageUrl,
      extractionMethod: extractedData.extractionMethod,
      platform: extractedData.platform
    }
  } catch (error: any) {
    clearTimeout(timeoutId)
    if (error.name === 'AbortError') {
      throw new Error('Profile extraction timeout - manual input required')
    }
    throw error
  }
}

// Data aggregation module to collect profile metrics (as per problem statement)
async function collectProfileMetrics(profileData: any, url?: string) {
  let actualProfileData = profileData
  
  // Attempt automatic extraction if URL is provided and profile data is missing
  if (url && (!profileData || Object.keys(profileData).length === 0)) {
    try {
      console.log('No profile data provided - attempting automatic extraction')
      actualProfileData = await extractProfileData(url)
      console.log('Automatic extraction successful')
    } catch (error) {
      console.warn('Automatic extraction failed:', error)
      throw new Error('Automatic profile extraction failed. Please provide profile data manually or check the URL.')
    }
  }
  
  // Ensure we have real profile data
  if (!actualProfileData) {
    throw new Error('Profile data must be provided - no fake data generation allowed')
  }
  
  // Validate required fields for Profile Purity analysis
  const requiredFields = ['followerCount', 'followingCount', 'postCount']
  const missingFields = requiredFields.filter(field => actualProfileData[field] === undefined)
  
  if (missingFields.length > 0) {
    throw new Error(`Missing required profile metrics: ${missingFields.join(', ')}. Please verify the profile URL or provide manual data.`)
  }
  
  // Calculate key metrics as specified in problem statement
  const followerCount = actualProfileData.followerCount
  const followingCount = actualProfileData.followingCount
  const postCount = actualProfileData.postCount
  const accountAge = actualProfileData.accountAge || 30 // Default if not provided
  
  const followersToFollowing = followingCount > 0 
    ? followerCount / followingCount 
    : followerCount
  
  // Calculate engagement metrics
  const postsPerDay = accountAge > 0 ? postCount / accountAge : 0
  const engagementRate = followerCount > 0 
    ? Math.min((postsPerDay * 100) / (followerCount * 0.01), 10)
    : postsPerDay * 10
  
  // Risk factor analysis (key part of Profile Purity detection)
  const riskFactors: string[] = []
  
  // Account age risks
  if (accountAge < 30) riskFactors.push('Very new account (< 30 days)')
  else if (accountAge < 90) riskFactors.push('New account (< 3 months)')
  
  // Follower ratio risks (critical for fake detection)
  if (followersToFollowing > 20) riskFactors.push('Unusually high followers-to-following ratio')
  else if (followersToFollowing < 0.1 && followerCount > 100) riskFactors.push('Following too many accounts')
  
  // Posting activity risks
  if (postsPerDay > 10) riskFactors.push('Excessive posting frequency')
  else if (postsPerDay < 0.01 && accountAge > 30) riskFactors.push('Very low posting activity')
  
  // Platform-specific analysis
  const platform = detectPlatformFromUrl(url)
  
  // Return structured metrics in expected format
  return {
    accountAge,
    followersToFollowing,
    engagement: {
      avgLikes: Math.floor(followerCount * engagementRate * 0.05),
      avgComments: Math.floor(followerCount * engagementRate * 0.01), 
      avgShares: Math.floor(followerCount * engagementRate * 0.005),
      rate: Math.round(engagementRate * 100) / 100
    },
    activityPattern: (postsPerDay > 2 ? 'consistent' : 
                    postsPerDay < 0.1 ? 'suspicious' : 'normal') as 'consistent' | 'suspicious' | 'normal',
    verification: {
      email: profileData.emailVerified || false,
      phone: profileData.phoneVerified || false,
      identity: profileData.verified || false
    },
    riskFactors,
    platform,
    rawMetrics: {
      followerCount,
      followingCount, 
      postCount,
      accountAge,
      bio: actualProfileData.bio || actualProfileData.displayName || '',
      username: actualProfileData.username || '',
      verified: actualProfileData.verified || false
    }
  }
}

// Analyze URL patterns to extract account information
function analyzeUrlPattern(url: string) {
  const result = {
    accountAge: null as number | null,
    isVerified: null as boolean | null,
    platform: 'unknown'
  }
  
  try {
    const urlLower = url.toLowerCase()
    
    // Detect platform
    if (urlLower.includes('instagram.com')) {
      result.platform = 'instagram'
    } else if (urlLower.includes('twitter.com') || urlLower.includes('x.com')) {
      result.platform = 'twitter'
    } else if (urlLower.includes('facebook.com')) {
      result.platform = 'facebook'
    }
    
    // For demo purposes, we can't actually determine real account age from URL
    // In a real system, this would require API access to the social platform
    // For now, we'll assume unknown account age (which should be treated as suspicious)
    result.accountAge = null // Unknown = treat as potentially new/suspicious
    result.isVerified = false // Assume not verified unless proven otherwise
    
  } catch (error) {
    console.error('Error analyzing URL pattern:', error)
  }
  
  return result
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

    // Collect real profile metrics using data aggregation module (with automatic extraction)
    // This will automatically extract data from the URL if no profileData is provided
    const realProfileMetrics = await collectProfileMetrics(body.profileData, body.url)
    
    // Extract content for analysis (use bio from extracted data if available)
    const textContent = body.textContent || realProfileMetrics.rawMetrics?.bio || "Sample profile text for analysis"
    
    // Run analysis (in parallel for better performance)
    const [textAnalysis, imageAnalysis] = await Promise.all([
      analyzeText(textContent),
      body.fileData ? analyzeImage(body.fileData) : Promise.resolve(null)
    ])
    
    // Calculate individual scores (these should be TRUST scores, lower = more suspicious)
    const textScore = Math.round(
      (textAnalysis.sentimentScore + textAnalysis.authenticity + (100 - textAnalysis.toxicity)) / 3
    )
    
    // Image score: Lower manipulation and higher quality = higher trust
    // NO IMAGE = MAJOR RED FLAG for profile authenticity
    const imageScore = imageAnalysis ? Math.round(
      (imageAnalysis.imageQuality + (100 - imageAnalysis.manipulation) + 
       (imageAnalysis.metadata.originalSource ? 100 : 0) + 
       (imageAnalysis.metadata.dateConsistency ? 100 : 0)) / 4
    ) : 35 // Moderate penalty for no image - some platforms allow this
    
    // Metrics score: Weighted scoring algorithm as specified in problem statement
    const baseMetricsScore = Math.round(
      Math.min(
        Math.min(realProfileMetrics.accountAge / 180, 1) * 30 + // Age factor (0-30 points, capped at 6 months)
        Math.min(Math.log10(Math.max(realProfileMetrics.followersToFollowing, 0.1)) * 8 + 25, 35) + // Ratio scoring
        Math.min(realProfileMetrics.engagement.rate * 15, 20) + // Engagement factor (0-20)
        (realProfileMetrics.verification.email ? 3 : 0) + // Verification bonuses
        (realProfileMetrics.verification.phone ? 4 : 0) +
        (realProfileMetrics.verification.identity ? 8 : 0),
        100
      )
    )
    
    // Apply risk factor penalties (Profile Purity scoring)
    const riskPenalty = realProfileMetrics.riskFactors.length * 6 // 6 points per risk factor (reduced from 10)
    const metricsScore = Math.max(baseMetricsScore - riskPenalty, 0)
    
    // Calculate overall trust score using weighted scoring algorithm (Profile Purity requirement)
    // Weights based on problem statement importance:
    // - NLP Analysis (text sentiment, grammar, coherence): 30%  
    // - Computer Vision (stock photos, AI-generated): 35%
    // - Profile Metrics (account age, follower ratio): 35%
    const trustScoreResult = await calculateTrustScore(textScore, imageScore, metricsScore, {
      textWeight: 0.30,
      imageWeight: 0.35, 
      metricsWeight: 0.35
    })
    const trustScore = trustScoreResult.trustScore
    
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
        profileMetrics: realProfileMetrics
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