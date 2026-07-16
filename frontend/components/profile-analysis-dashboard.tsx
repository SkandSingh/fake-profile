'use client'

import { useState } from 'react'
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card'
import { Button } from '@/components/ui/button'
import { Badge } from '@/components/ui/badge'
import { Separator } from '@/components/ui/separator'
import ProfileInputForm from './profile-input-form'
import { UrlExtractForm, type ExtractedProfileData } from './url-extract-form'
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
  inputType: 'url' | 'manual'
  inputValue: string
  trustScore: number
  status: 'completed' | 'processing' | 'failed'
  apiResult: any
}

export function ProfileAnalysisDashboard() {
  const [currentAnalysis, setCurrentAnalysis] = useState<AnalysisResult | null>(null)
  const [isAnalyzing, setIsAnalyzing] = useState(false)
  const [analysisHistory, setAnalysisHistory] = useState<AnalysisResult[]>([])
  const [activeTab, setActiveTab] = useState<'url' | 'manual'>('url')
  const [prefillData, setPrefillData] = useState<Partial<{
    username: string; displayName: string; platform: string; bio: string
    followerCount: number; followingCount: number; postCount: number
  }> | undefined>(undefined)
  const [extractNotice, setExtractNotice] = useState<string | null>(null)

  const handleUrlExtracted = (data: ExtractedProfileData) => {
    setPrefillData({
      username: data.username,
      displayName: data.displayName,
      platform: data.platform === 'unknown' ? undefined : data.platform,
      bio: data.bio,
      followerCount: data.followerCount,
      followingCount: data.followingCount,
      postCount: data.postCount,
    })
    setExtractNotice(
      data.manualInputRequired
        ? `Auto-extracted what we could${data.extractionError ? ` (${data.extractionError})` : ''}. Please fill in or confirm the rest: ${data.missingFields.join(', ') || 'remaining fields'}.`
        : 'Auto-extracted successfully - review and submit below.'
    )
    setActiveTab('manual')
  }

  const handleAnalysisComplete = (apiResult: any) => {
    const result: AnalysisResult = {
      id: `analysis-${Date.now()}`,
      timestamp: new Date(),
      inputType: prefillData ? 'url' : 'manual',
      inputValue: apiResult.profileSummary?.username || 'Manual Input',
      trustScore: apiResult.trustScore,
      status: 'completed',
      apiResult
    }

    setCurrentAnalysis(result)
    setAnalysisHistory(prev => [result, ...prev.slice(0, 4)]) // Keep last 5 analyses
    setIsAnalyzing(false)
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
    setActiveTab('url')
    setPrefillData(undefined)
    setExtractNotice(null)
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
    <div className="min-h-screen bg-gradient-to-br from-blue-50 to-indigo-100">
      <div className="container mx-auto px-4 py-6 md:py-8">
        {/* Header */}
        <div className="mb-8">
          <div className="flex items-center justify-between mb-4">
            <div className="flex items-center gap-3">
              <div className="p-2 bg-blue-600 rounded-lg">
                <Shield className="h-6 w-6 text-white" />
              </div>
              <div>
                <h1 className="text-2xl md:text-3xl font-bold text-gray-900">
                  Profile Purity Detector
                </h1>
                <p className="text-sm md:text-base text-gray-600">
                  Automatic profile extraction + AI-powered fake detection
                </p>
              </div>
            </div>
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
                  {activeTab === 'url' ? 'Paste URL & Auto-Analyze' : 'Manual Profile Analysis'}
                </CardTitle>
                <div className="flex gap-2 pt-2">
                  <Button
                    type="button"
                    size="sm"
                    variant={activeTab === 'url' ? 'default' : 'outline'}
                    onClick={() => setActiveTab('url')}
                  >
                    Paste URL
                  </Button>
                  <Button
                    type="button"
                    size="sm"
                    variant={activeTab === 'manual' ? 'default' : 'outline'}
                    onClick={() => setActiveTab('manual')}
                  >
                    Manual Entry
                  </Button>
                </div>
              </CardHeader>
              <CardContent>
                {activeTab === 'url' ? (
                  <UrlExtractForm onExtracted={handleUrlExtracted} />
                ) : (
                  <>
                    {extractNotice && (
                      <div className="mb-4 p-3 rounded-md bg-blue-50 text-sm text-blue-800">{extractNotice}</div>
                    )}
                    <ProfileInputForm onAnalysisComplete={handleAnalysisComplete} initialData={prefillData} />
                  </>
                )}
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
                        data={{
                          id: currentAnalysis.id,
                          timestamp: currentAnalysis.timestamp,
                          inputType: currentAnalysis.inputType,
                          inputValue: currentAnalysis.inputValue,
                          trustScore: currentAnalysis.trustScore,
                          textAnalysis: currentAnalysis.apiResult.breakdown.textAnalysis,
                          imageAnalysis: currentAnalysis.apiResult.breakdown.imageAnalysis,
                          profileMetrics: currentAnalysis.apiResult.breakdown.profileMetrics,
                        }}
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

            {/* Enhanced Results Display - values come straight from the API response (real metrics-model + Gemini output), nothing re-derived or randomized on the client */}
            {currentAnalysis && !isAnalyzing && (
              <EnhancedResultsPage
                result={{
                  trustScore: currentAnalysis.trustScore,
                  textScore: currentAnalysis.apiResult.textScore,
                  imageScore: currentAnalysis.apiResult.imageScore,
                  metricsScore: currentAnalysis.apiResult.metricsScore,
                  explanation: currentAnalysis.apiResult.explanation,
                  breakdown: currentAnalysis.apiResult.breakdown,
                  processingTime: currentAnalysis.apiResult.processingTimeMs ?? 0,
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