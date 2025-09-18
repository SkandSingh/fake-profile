import { NextResponse } from 'next/server'

interface AnalysisStats {
  totalAnalyses: number
  last24Hours: number
  averageTrustScore: number
  riskDistribution: {
    highRisk: number // < 40
    moderateRisk: number // 40-69
    lowRisk: number // >= 70
  }
  popularInputTypes: {
    url: number
    file: number
  }
  serviceUptime: number
  lastUpdated: string
}

// Mock data storage (in production, this would be a database)
let mockStats = {
  totalAnalyses: 1247,
  recentAnalyses: [] as Array<{
    timestamp: Date
    trustScore: number
    inputType: 'url' | 'file'
  }>,
  serviceStartTime: Date.now()
}

// Simulate some historical data
for (let i = 0; i < 100; i++) {
  const hoursAgo = Math.random() * 48 // Last 48 hours
  mockStats.recentAnalyses.push({
    timestamp: new Date(Date.now() - hoursAgo * 60 * 60 * 1000),
    trustScore: Math.floor(Math.random() * 100),
    inputType: Math.random() > 0.6 ? 'url' : 'file'
  })
}

export async function GET() {
  try {
    const now = new Date()
    const twentyFourHoursAgo = new Date(now.getTime() - 24 * 60 * 60 * 1000)
    
    // Filter analyses from last 24 hours
    const last24HourAnalyses = mockStats.recentAnalyses.filter(
      analysis => analysis.timestamp >= twentyFourHoursAgo
    )
    
    // Calculate average trust score
    const allScores = mockStats.recentAnalyses.map(a => a.trustScore)
    const averageTrustScore = allScores.length > 0
      ? Math.round(allScores.reduce((sum, score) => sum + score, 0) / allScores.length)
      : 0
    
    // Calculate risk distribution
    const highRisk = allScores.filter(score => score < 40).length
    const moderateRisk = allScores.filter(score => score >= 40 && score < 70).length
    const lowRisk = allScores.filter(score => score >= 70).length
    
    // Calculate input type distribution
    const urlAnalyses = mockStats.recentAnalyses.filter(a => a.inputType === 'url').length
    const fileAnalyses = mockStats.recentAnalyses.filter(a => a.inputType === 'file').length
    
    // Calculate service uptime
    const serviceUptime = ((Date.now() - mockStats.serviceStartTime) / (24 * 60 * 60 * 1000)) * 99.5 // Mock 99.5% uptime
    
    const stats: AnalysisStats = {
      totalAnalyses: mockStats.totalAnalyses + mockStats.recentAnalyses.length,
      last24Hours: last24HourAnalyses.length,
      averageTrustScore,
      riskDistribution: {
        highRisk,
        moderateRisk,
        lowRisk
      },
      popularInputTypes: {
        url: urlAnalyses,
        file: fileAnalyses
      },
      serviceUptime: Math.min(serviceUptime, 100),
      lastUpdated: now.toISOString()
    }
    
    return NextResponse.json(stats)
    
  } catch (error) {
    console.error('Stats error:', error)
    return NextResponse.json(
      { error: 'Failed to fetch analysis statistics' },
      { status: 500 }
    )
  }
}

// Handle POST to add a new analysis to stats
export async function POST(request: Request) {
  try {
    const { trustScore, inputType } = await request.json()
    
    if (typeof trustScore !== 'number' || !['url', 'file'].includes(inputType)) {
      return NextResponse.json(
        { error: 'Invalid data: trustScore (number) and inputType (url|file) required' },
        { status: 400 }
      )
    }
    
    // Add to mock data
    mockStats.recentAnalyses.push({
      timestamp: new Date(),
      trustScore,
      inputType
    })
    
    // Keep only recent analyses (last 1000)
    if (mockStats.recentAnalyses.length > 1000) {
      mockStats.recentAnalyses = mockStats.recentAnalyses.slice(-1000)
    }
    
    return NextResponse.json({ success: true })
    
  } catch (error) {
    console.error('Stats update error:', error)
    return NextResponse.json(
      { error: 'Failed to update statistics' },
      { status: 500 }
    )
  }
}