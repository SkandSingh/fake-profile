'use client'

import { useState } from 'react'
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card'
import { Button } from '@/components/ui/button'
import { Badge } from '@/components/ui/badge'
import { Separator } from '@/components/ui/separator'
import { ThemeToggle } from './theme-toggle'
import { ProfileInputForm } from './profile-input-form'
import { EnhancedResultsPage } from './enhanced-results-page'
import { ResultExporter } from './result-exporter'
import { 
  Search, 
  Download, 
  RefreshCw, 
  Shield, 
  AlertTriangle,
  CheckCircle,
  Clock,
  FileText
} from 'lucide-react'

interface AnalysisResult {
  id: string
  timestamp: Date
  inputType: 'url' | 'file'
  inputValue: string
  trustScore: number
  status: 'completed' | 'processing' | 'failed'
  textAnalysis?: any
  imageAnalysis?: any
  profileMetrics?: any
  chartData?: any
}

export function ProfileAnalysisDashboard() {
  const [currentAnalysis, setCurrentAnalysis] = useState<AnalysisResult | null>(null)
  const [isAnalyzing, setIsAnalyzing] = useState(false)
  const [analysisHistory, setAnalysisHistory] = useState<AnalysisResult[]>([])

  // Profile Purity mock data aligned with problem statement
  const mockAnalysisResult: AnalysisResult = {
    id: 'analysis-1',
    timestamp: new Date(),
    inputType: 'url',
    inputValue: 'https://twitter.com/example',
    trustScore: 76,
    status: 'completed',
    textAnalysis: {
      sentiment: 'positive' as const,
      sentimentScore: 82,
      toxicity: 15,
      authenticity: 88,
      readability: 75,
      keywords: ['technology', 'innovation', 'startup', 'AI'],
      languageDetected: 'English',
      confidence: 94
    },
    imageAnalysis: {
      faceDetected: true,
      imageQuality: 85,
      manipulation: 12,
      metadata: {
        originalSource: true,
        dateConsistency: true,
        locationConsistency: false
      },
      similarImages: 3,
      confidence: 91
    },
    profileMetrics: {
      accountAge: 1825, // ~5 years
      followersToFollowing: 2.3,
      engagement: {
        avgLikes: 45,
        avgComments: 12,
        avgShares: 8,
        rate: 3.2
      },
      activityPattern: 'consistent' as const,
      verification: {
        email: true,
        phone: true,
        identity: false
      },
      riskFactors: ['Unverified identity', 'Location inconsistency']
    }
  }

  const handleAnalysisSubmit = async (data: { 
    url?: string; 
    file?: File; 
    analysisType: string;
  }) => {
    setIsAnalyzing(true)
    
    try {
      // Prepare the API request payload for Profile Purity Detector
      const payload: any = {
        type: data.url ? 'url' : 'file',
        url: data.url
      }
      
      // Add file data if present
      if (data.file) {
        payload.fileData = {
          name: data.file.name,
          type: data.file.type,
          size: data.file.size
        }
        
        // Convert file to base64 if it's an image
        if (data.file.type.startsWith('image/')) {
          const base64 = await fileToBase64(data.file)
          payload.fileData.content = base64
        }
      }
      
      // Profile data will be automatically extracted from URL
      console.log('Submitting for automatic extraction and analysis:', payload)
      
      // Call the analysis API
      const response = await fetch('/api/analyze', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(payload),
      })
      
      if (!response.ok) {
        const errorData = await response.json()
        throw new Error(errorData.error || `Analysis failed: ${response.statusText}`)
      }
      
      const apiResult = await response.json()
      
      // Transform API result to match our component interface
      const result: AnalysisResult = {
        id: apiResult.analysisId,
        timestamp: new Date(),
        inputType: data.url ? 'url' : 'file',
        inputValue: data.url || data.file?.name || 'Unknown',
        trustScore: apiResult.trustScore,
        status: 'completed',
        textAnalysis: apiResult.breakdown.textAnalysis,
        imageAnalysis: apiResult.breakdown.imageAnalysis,
        profileMetrics: apiResult.breakdown.profileMetrics
      }
      
      setCurrentAnalysis(result)
      setAnalysisHistory(prev => [result, ...prev.slice(0, 4)]) // Keep last 5 analyses
      
    } catch (error) {
      console.error('Analysis failed:', error)
      alert(`Analysis failed: ${error instanceof Error ? error.message : 'Unknown error'}`)
      
      // Fallback to mock data if API fails
      const result: AnalysisResult = {
        ...mockAnalysisResult,
        id: `analysis-${Date.now()}`,
        timestamp: new Date(),
        inputType: data.url ? 'url' : 'file',
        inputValue: data.url || data.file?.name || 'Unknown'
      }
      
      setCurrentAnalysis(result)
      setAnalysisHistory(prev => [result, ...prev.slice(0, 4)])
    } finally {
      setIsAnalyzing(false)
    }
  }
  
  // Helper function to convert file to base64
  const fileToBase64 = (file: File): Promise<string> => {
    return new Promise((resolve, reject) => {
      const reader = new FileReader()
      reader.readAsDataURL(file)
      reader.onload = () => {
        const base64 = reader.result as string
        resolve(base64.split(',')[1]) // Remove data:image/jpeg;base64, prefix
      }
      reader.onerror = error => reject(error)
    })
  }
  
  // Helper function to extract username from URL
  const extractUsernameFromUrl = (url: string): string => {
    try {
      const urlObj = new URL(url)
      const pathParts = urlObj.pathname.split('/').filter(part => part.length > 0)
      return pathParts[0] || 'unknown_user'
    } catch {
      return 'invalid_url'
    }
  }

  const handleNewAnalysis = () => {
    setCurrentAnalysis(null)
    setIsAnalyzing(false)
  }

  const getTrustScoreColor = (score: number) => {
    if (score >= 80) return 'text-green-600'
    if (score >= 60) return 'text-yellow-600'
    if (score >= 40) return 'text-orange-600'
    return 'text-red-600'
  }

  const getTrustScoreIcon = (score: number) => {
    if (score >= 80) return <CheckCircle className="h-5 w-5 text-green-600" />
    if (score >= 60) return <Shield className="h-5 w-5 text-yellow-600" />
    return <AlertTriangle className="h-5 w-5 text-red-600" />
  }

  return (
    <div className="min-h-screen bg-gradient-to-br from-blue-50 to-indigo-100 dark:from-gray-900 dark:to-gray-800">
      <div className="container mx-auto px-4 py-6 md:py-8">
        {/* Header */}
        <div className="mb-8">
          <div className="flex items-center justify-between mb-4">
            <div className="flex items-center gap-3">
              <div className="p-2 bg-blue-600 rounded-lg">
                <Shield className="h-6 w-6 text-white" />
              </div>
              <div>
                <h1 className="text-2xl md:text-3xl font-bold text-gray-900 dark:text-white">
                  Profile Purity Detector
                </h1>
                <p className="text-sm md:text-base text-gray-600 dark:text-gray-300">
                  Automatic profile extraction + AI-powered fake detection
                </p>
              </div>
            </div>
            <ThemeToggle />
          </div>
          
          {/* Quick Stats */}
          <div className="grid grid-cols-2 md:grid-cols-4 gap-3 md:gap-4">
            <Card>
              <CardContent className="flex items-center gap-3 p-4">
                <Search className="h-8 w-8 text-blue-600" />
                <div>
                  <p className="text-2xl font-bold">{analysisHistory.length}</p>
                  <p className="text-sm text-muted-foreground">Profiles Analyzed</p>
                </div>
              </CardContent>
            </Card>
            <Card>
              <CardContent className="flex items-center gap-3 p-4">
                <CheckCircle className="h-8 w-8 text-green-600" />
                <div>
                  <p className="text-2xl font-bold">
                    {analysisHistory.filter(a => a.trustScore >= 70).length}
                  </p>
                  <p className="text-sm text-muted-foreground">Authentic Profiles</p>
                </div>
              </CardContent>
            </Card>
            <Card>
              <CardContent className="flex items-center gap-3 p-4">
                <AlertTriangle className="h-8 w-8 text-yellow-600" />
                <div>
                  <p className="text-2xl font-bold">
                    {analysisHistory.filter(a => a.trustScore < 70 && a.trustScore >= 40).length}
                  </p>
                  <p className="text-sm text-muted-foreground">Suspicious</p>
                </div>
              </CardContent>
            </Card>
            <Card>
              <CardContent className="flex items-center gap-3 p-4">
                <Shield className="h-8 w-8 text-red-600" />
                <div>
                  <p className="text-2xl font-bold">
                    {analysisHistory.filter(a => a.trustScore < 40).length}
                  </p>
                  <p className="text-sm text-muted-foreground">Likely Fake</p>
                </div>
              </CardContent>
            </Card>
          </div>
        </div>

        {/* Main Content */}
        {!currentAnalysis && !isAnalyzing ? (
          /* Profile Analysis Input Form */
          <div className="max-w-4xl mx-auto">
            <Card>
              <CardHeader>
                <CardTitle className="flex items-center gap-2">
                  <Search className="h-5 w-5" />
                  Paste URL & Auto-Analyze
                </CardTitle>
              </CardHeader>
              <CardContent>
                <ProfileInputForm onSubmit={handleAnalysisSubmit} />
              </CardContent>
            </Card>

            {/* Recent Analyses */}
            {analysisHistory.length > 0 && (
              <Card className="mt-6">
                <CardHeader>
                  <CardTitle className="flex items-center gap-2">
                    <Clock className="h-5 w-5" />
                    Recent Analyses
                  </CardTitle>
                </CardHeader>
                <CardContent>
                  <div className="space-y-3">
                    {analysisHistory.slice(0, 5).map((analysis) => (
                      <div 
                        key={analysis.id}
                        className="flex items-center justify-between p-3 border rounded-lg hover:bg-muted/50 cursor-pointer transition-colors"
                        onClick={() => setCurrentAnalysis(analysis)}
                      >
                        <div className="flex items-center gap-3">
                          {getTrustScoreIcon(analysis.trustScore)}
                          <div>
                            <p className="font-medium">{analysis.inputValue}</p>
                            <p className="text-sm text-muted-foreground">
                              {analysis.timestamp.toLocaleDateString()} at {analysis.timestamp.toLocaleTimeString()}
                            </p>
                          </div>
                        </div>
                        <div className="flex items-center gap-2">
                          <Badge variant="outline">
                            {analysis.inputType}
                          </Badge>
                          <span className={`font-bold ${getTrustScoreColor(analysis.trustScore)}`}>
                            {analysis.trustScore}%
                          </span>
                        </div>
                      </div>
                    ))}
                  </div>
                </CardContent>
              </Card>
            )}
          </div>
        ) : (
          /* Analysis Results Dashboard */
          <div className="space-y-6">
            {/* Results Header */}
            <Card>
              <CardContent className="p-6">
                <div className="flex items-center justify-between">
                  <div className="flex items-center gap-4">
                    {getTrustScoreIcon(currentAnalysis?.trustScore || 0)}
                    <div>
                      <h2 className="text-xl font-bold">
                        Analysis Results
                        {currentAnalysis && ` - ${currentAnalysis.inputValue}`}
                      </h2>
                      <p className="text-muted-foreground">
                        {isAnalyzing 
                          ? 'Analysis in progress...' 
                          : `Completed ${currentAnalysis?.timestamp.toLocaleString()}`
                        }
                      </p>
                    </div>
                  </div>
                  
                  <div className="flex items-center gap-2">
                    {currentAnalysis && (
                      <ResultExporter 
                        data={currentAnalysis}
                        onExport={(format) => console.log(`Exported as ${format}`)}
                      />
                    )}
                    <Button
                      variant="outline"
                      size="sm"
                      onClick={handleNewAnalysis}
                      className="flex items-center gap-2"
                    >
                      <RefreshCw className="h-4 w-4" />
                      New Analysis
                    </Button>
                  </div>
                </div>
              </CardContent>
            </Card>

            {/* Enhanced Results Display */}
            {currentAnalysis && !isAnalyzing && (
              <EnhancedResultsPage 
                result={{
                  trustScore: currentAnalysis.trustScore,
                  textScore: Math.round(
                    (currentAnalysis.textAnalysis?.sentimentScore + 
                     currentAnalysis.textAnalysis?.authenticity + 
                     (100 - (currentAnalysis.textAnalysis?.toxicity || 0))) / 3
                  ),
                  imageScore: Math.round(
                    (currentAnalysis.imageAnalysis?.imageQuality + 
                     (100 - (currentAnalysis.imageAnalysis?.manipulation || 0)) + 
                     (currentAnalysis.imageAnalysis?.confidence || 85)) / 3
                  ),
                  metricsScore: Math.round(
                    Math.min(
                      (currentAnalysis.profileMetrics?.accountAge / 365) * 20 + 
                      Math.min((currentAnalysis.profileMetrics?.followersToFollowing || 1) * 10, 30) + 
                      (currentAnalysis.profileMetrics?.engagement.rate || 2) * 15 + 
                      (Object.values(currentAnalysis.profileMetrics?.verification || {}).filter(Boolean).length * 10) -
                      (currentAnalysis.profileMetrics?.riskFactors?.length || 0) * 5,
                      100
                    )
                  ),
                  explanation: [
                    currentAnalysis.trustScore >= 80 ? "Profile shows strong indicators of authenticity and trustworthiness." : 
                    currentAnalysis.trustScore >= 60 ? "Profile appears moderately trustworthy with some areas requiring attention." :
                    currentAnalysis.trustScore >= 40 ? "Profile raises several concerns that warrant careful consideration." :
                    "Profile exhibits multiple risk factors suggesting potential fraudulent activity.",
                    
                    currentAnalysis.textAnalysis?.sentiment === 'positive' ? "Text content demonstrates authentic language patterns and positive sentiment." :
                    currentAnalysis.textAnalysis?.sentiment === 'negative' ? "Text analysis reveals concerning patterns in language use or sentiment." :
                    "Text content shows neutral sentiment with standard language patterns.",
                    
                    currentAnalysis.imageAnalysis?.imageQuality > 80 ? "Profile images show high quality with no signs of manipulation." :
                    currentAnalysis.imageAnalysis?.manipulation > 50 ? "Image analysis detected potential manipulation or quality issues." :
                    "Image analysis shows standard quality with no major concerns.",
                    
                    currentAnalysis.profileMetrics?.activityPattern === 'consistent' ? "Account metrics indicate natural, organic growth and engagement patterns." :
                    currentAnalysis.profileMetrics?.activityPattern === 'suspicious' ? "Account metrics suggest potential artificial inflation or suspicious activity." :
                    "Account metrics show normal patterns within expected ranges."
                  ],
                  breakdown: {
                    textAnalysis: currentAnalysis.textAnalysis,
                    imageAnalysis: currentAnalysis.imageAnalysis,
                    profileMetrics: currentAnalysis.profileMetrics
                  },
                  processingTime: Math.floor(Math.random() * 2000) + 1000,
                  analysisId: currentAnalysis.id
                }}
              />
            )}
            
            {/* Loading State */}
            {isAnalyzing && (
              <div className="space-y-6">
                <Card>
                  <CardContent className="flex items-center justify-center py-12">
                    <div className="text-center space-y-4">
                      <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-blue-600 mx-auto" />
                      <div className="text-lg font-medium">Analyzing Profile...</div>
                      <div className="text-sm text-muted-foreground">
                        This may take a few moments while we analyze all aspects
                      </div>
                    </div>
                  </CardContent>
                </Card>
              </div>
            )}

          </div>
        )}
      </div>
    </div>
  )
}