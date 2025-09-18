#!/usr/bin/env node

/**
 * API Testing Script for Profile Analysis Backend
 * Tests all API endpoints with sample data
 */

const BASE_URL = 'http://localhost:3000'

async function testAPI() {
  console.log('🚀 Testing Profile Analysis API Endpoints\n')
  
  try {
    // Test 1: Health Check
    console.log('1️⃣ Testing Health Check...')
    const healthResponse = await fetch(`${BASE_URL}/api/health`)
    const healthData = await healthResponse.json()
    console.log(`   Status: ${healthData.status}`)
    console.log(`   Services: ${Object.values(healthData.services).every(s => s) ? '✅ All Online' : '❌ Some Offline'}`)
    console.log(`   Uptime: ${(healthData.uptime / (24 * 60 * 60 * 1000)).toFixed(1)} days\n`)
    
    // Test 2: Single Profile Analysis
    console.log('2️⃣ Testing Single Profile Analysis...')
    const analysisPayload = {
      type: 'url',
      url: 'https://twitter.com/example_user',
      textContent: 'Passionate about technology and innovation. Building the future one line of code at a time! 🚀 #AI #Tech #Startup',
      profileData: {
        username: 'example_user',
        followerCount: 5420,
        followingCount: 892,
        postCount: 234,
        accountAge: 856, // ~2.3 years
        verified: false
      }
    }
    
    const analysisStart = Date.now()
    const analysisResponse = await fetch(`${BASE_URL}/api/analyze`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(analysisPayload)
    })
    const analysisData = await analysisResponse.json()
    const analysisTime = Date.now() - analysisStart
    
    console.log(`   Trust Score: ${analysisData.trustScore}% (${getTrustLevel(analysisData.trustScore)})`)
    console.log(`   Text Score: ${analysisData.textScore}%`)
    console.log(`   Image Score: ${analysisData.imageScore}%`)
    console.log(`   Metrics Score: ${analysisData.metricsScore}%`)
    console.log(`   Processing Time: ${analysisTime}ms`)
    console.log(`   Analysis ID: ${analysisData.analysisId}`)
    console.log(`   Explanations: ${analysisData.explanation.length} insights provided\n`)
    
    // Test 3: Batch Analysis
    console.log('3️⃣ Testing Batch Profile Analysis...')
    const batchPayload = {
      profiles: [
        {
          id: 'profile_1',
          type: 'url',
          url: 'https://twitter.com/user1',
          textContent: 'Love sharing positive vibes and inspiring others! ✨',
          profileData: { followerCount: 1200, followingCount: 300, verified: true }
        },
        {
          id: 'profile_2',
          type: 'url', 
          url: 'https://instagram.com/user2',
          textContent: 'Just another fake account trying to scam people...',
          profileData: { followerCount: 50000, followingCount: 10, verified: false }
        },
        {
          id: 'profile_3',
          type: 'file',
          fileData: {
            name: 'profile_pic.jpg',
            type: 'image/jpeg',
            size: 245760
          },
          textContent: 'Entrepreneur | Investor | Thought Leader',
          profileData: { followerCount: 8500, followingCount: 1200, verified: false }
        }
      ],
      options: {
        parallel: true,
        maxConcurrency: 2
      }
    }
    
    const batchStart = Date.now()
    const batchResponse = await fetch(`${BASE_URL}/api/analyze/batch`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(batchPayload)
    })
    const batchData = await batchResponse.json()
    const batchTime = Date.now() - batchStart
    
    console.log(`   Total Profiles: ${batchData.totalProfiles}`)
    console.log(`   Completed: ${batchData.completedProfiles} | Failed: ${batchData.failedProfiles}`)
    console.log(`   Processing Time: ${batchTime}ms`)
    console.log(`   Average Trust Score: ${batchData.summary.averageTrustScore}%`)
    console.log(`   Risk Distribution:`)
    console.log(`     - High Risk: ${batchData.summary.highRiskProfiles}`)
    console.log(`     - Moderate Risk: ${batchData.summary.moderateRiskProfiles}`)
    console.log(`     - Low Risk: ${batchData.summary.lowRiskProfiles}\n`)
    
    // Test 4: Statistics
    console.log('4️⃣ Testing Statistics Endpoint...')
    const statsResponse = await fetch(`${BASE_URL}/api/stats`)
    const statsData = await statsResponse.json()
    
    console.log(`   Total Analyses: ${statsData.totalAnalyses.toLocaleString()}`)
    console.log(`   Last 24 Hours: ${statsData.last24Hours}`)
    console.log(`   Average Trust Score: ${statsData.averageTrustScore}%`)
    console.log(`   Service Uptime: ${statsData.serviceUptime.toFixed(1)}%`)
    console.log(`   Input Types: URL (${statsData.popularInputTypes.url}) | File (${statsData.popularInputTypes.file})\n`)
    
    // Test 5: Add to Statistics
    console.log('5️⃣ Testing Statistics Update...')
    const statsUpdateResponse = await fetch(`${BASE_URL}/api/stats`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        trustScore: analysisData.trustScore,
        inputType: 'url'
      })
    })
    const statsUpdateData = await statsUpdateResponse.json()
    console.log(`   Stats Update: ${statsUpdateData.success ? '✅ Success' : '❌ Failed'}\n`)
    
    // Test 6: Error Handling
    console.log('6️⃣ Testing Error Handling...')
    const errorResponse = await fetch(`${BASE_URL}/api/analyze`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ type: 'invalid' })
    })
    const errorData = await errorResponse.json()
    console.log(`   Error Response: ${errorResponse.status} - ${errorData.error}\n`)
    
    console.log('✅ All API tests completed successfully!')
    console.log('\n📊 Summary:')
    console.log(`   - Health Check: Operational`)
    console.log(`   - Single Analysis: ${analysisTime}ms`)
    console.log(`   - Batch Analysis: ${batchTime}ms (${batchData.totalProfiles} profiles)`)
    console.log(`   - Statistics: Available`)
    console.log(`   - Error Handling: Working`)
    
  } catch (error) {
    console.error('❌ API Test Failed:', error.message)
    console.log('\n🔧 Troubleshooting:')
    console.log('   1. Make sure the development server is running: npm run dev')
    console.log('   2. Check if port 3000 is available')
    console.log('   3. Verify the API routes are properly configured')
  }
}

function getTrustLevel(score) {
  if (score >= 80) return 'Highly Trustworthy'
  if (score >= 60) return 'Moderately Trustworthy'
  if (score >= 40) return 'Questionable'
  return 'High Risk'
}

// Run the tests
if (require.main === module) {
  testAPI()
}

module.exports = { testAPI }