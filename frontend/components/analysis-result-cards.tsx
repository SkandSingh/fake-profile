'use client'

import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card'
import { Badge } from '@/components/ui/badge'
import { Progress } from '@/components/ui/progress'
import { 
  FileText, 
  Image, 
  User, 
  TrendingUp, 
  TrendingDown, 
  Minus,
  CheckCircle,
  XCircle,
  AlertCircle,
  Eye,
  MessageCircle,
  Heart,
  Share
} from 'lucide-react'

interface AnalysisMetric {
  label: string
  value: number | string
  trend?: 'up' | 'down' | 'neutral'
  status?: 'positive' | 'negative' | 'neutral'
}

interface TextAnalysisResult {
  sentiment: 'positive' | 'negative' | 'neutral'
  sentimentScore: number
  toxicity: number
  authenticity: number
  readability: number
  keywords: string[]
  languageDetected: string
  confidence: number
}

interface ImageAnalysisResult {
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

interface ProfileMetricsResult {
  accountAge: number // in days
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

interface AnalysisResultCardsProps {
  textAnalysis?: TextAnalysisResult
  imageAnalysis?: ImageAnalysisResult
  profileMetrics?: ProfileMetricsResult
  isLoading?: boolean
}

export function AnalysisResultCards({ 
  textAnalysis, 
  imageAnalysis, 
  profileMetrics, 
  isLoading = false 
}: AnalysisResultCardsProps) {
  
  const getSentimentColor = (sentiment: string) => {
    switch (sentiment) {
      case 'positive': return 'text-green-600'
      case 'negative': return 'text-red-600'
      default: return 'text-yellow-600'
    }
  }

  const getSentimentIcon = (sentiment: string) => {
    switch (sentiment) {
      case 'positive': return <CheckCircle className="h-4 w-4 text-green-600" />
      case 'negative': return <XCircle className="h-4 w-4 text-red-600" />
      default: return <AlertCircle className="h-4 w-4 text-yellow-600" />
    }
  }

  const getTrendIcon = (trend: string) => {
    switch (trend) {
      case 'up': return <TrendingUp className="h-4 w-4 text-green-600" />
      case 'down': return <TrendingDown className="h-4 w-4 text-red-600" />
      default: return <Minus className="h-4 w-4 text-gray-600" />
    }
  }

  const getScoreColor = (score: number) => {
    if (score >= 80) return 'text-green-600'
    if (score >= 60) return 'text-yellow-600'
    if (score >= 40) return 'text-orange-600'
    return 'text-red-600'
  }

  const getProgressColor = (score: number) => {
    if (score >= 80) return 'bg-green-500'
    if (score >= 60) return 'bg-yellow-500'
    if (score >= 40) return 'bg-orange-500'
    return 'bg-red-500'
  }

  const LoadingCard = ({ title, icon }: { title: string; icon: React.ReactNode }) => (
    <Card>
      <CardHeader>
        <CardTitle className="flex items-center gap-2">
          {icon}
          {title}
        </CardTitle>
      </CardHeader>
      <CardContent className="space-y-4">
        <div className="space-y-3">
          {[1, 2, 3].map((i) => (
            <div key={i} className="space-y-2">
              <div className="h-4 bg-gray-200 rounded animate-pulse" />
              <div className="h-2 bg-gray-100 rounded animate-pulse" />
            </div>
          ))}
        </div>
      </CardContent>
    </Card>
  )

  if (isLoading) {
    return (
      <div className="grid gap-6 md:grid-cols-3">
        <LoadingCard title="Text Analysis" icon={<FileText className="h-5 w-5" />} />
        <LoadingCard title="Image Analysis" icon={<Image className="h-5 w-5" />} />
        <LoadingCard title="Profile Metrics" icon={<User className="h-5 w-5" />} />
      </div>
    )
  }

  return (
    <div className="grid gap-6 md:grid-cols-3">
      {/* Text Analysis Card */}
      {textAnalysis && (
        <Card>
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              <FileText className="h-5 w-5" />
              Text Analysis
            </CardTitle>
          </CardHeader>
          <CardContent className="space-y-4">
            {/* Sentiment */}
            <div className="space-y-2">
              <div className="flex items-center justify-between">
                <span className="text-sm font-medium">Sentiment</span>
                <div className="flex items-center gap-1">
                  {getSentimentIcon(textAnalysis.sentiment)}
                  <span className={`text-sm font-medium ${getSentimentColor(textAnalysis.sentiment)}`}>
                    {textAnalysis.sentiment}
                  </span>
                </div>
              </div>
              <Progress 
                value={textAnalysis.sentimentScore} 
                className="h-2"
              />
              <span className="text-xs text-muted-foreground">
                Score: {textAnalysis.sentimentScore}%
              </span>
            </div>

            {/* Authenticity */}
            <div className="space-y-2">
              <div className="flex items-center justify-between">
                <span className="text-sm font-medium">Authenticity</span>
                <span className={`text-sm font-medium ${getScoreColor(textAnalysis.authenticity)}`}>
                  {textAnalysis.authenticity}%
                </span>
              </div>
              <Progress 
                value={textAnalysis.authenticity} 
                className="h-2"
              />
            </div>

            {/* Toxicity */}
            <div className="space-y-2">
              <div className="flex items-center justify-between">
                <span className="text-sm font-medium">Toxicity</span>
                <span className={`text-sm font-medium ${textAnalysis.toxicity > 20 ? 'text-red-600' : 'text-green-600'}`}>
                  {textAnalysis.toxicity}%
                </span>
              </div>
              <Progress 
                value={textAnalysis.toxicity} 
                className="h-2"
              />
            </div>

            {/* Keywords */}
            <div className="space-y-2">
              <span className="text-sm font-medium">Key Topics</span>
              <div className="flex flex-wrap gap-1">
                {textAnalysis.keywords.slice(0, 4).map((keyword, index) => (
                  <Badge key={index} variant="secondary" className="text-xs">
                    {keyword}
                  </Badge>
                ))}
              </div>
            </div>

            {/* Language */}
            <div className="flex items-center justify-between text-sm">
              <span className="text-muted-foreground">Language</span>
              <span className="font-medium">{textAnalysis.languageDetected}</span>
            </div>
          </CardContent>
        </Card>
      )}

      {/* Image Analysis Card */}
      {imageAnalysis && (
        <Card>
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              <Image className="h-5 w-5" />
              Image Analysis
            </CardTitle>
          </CardHeader>
          <CardContent className="space-y-4">
            {/* Face Detection */}
            <div className="flex items-center justify-between">
              <span className="text-sm font-medium">Face Detected</span>
              <div className="flex items-center gap-1">
                {imageAnalysis.faceDetected ? (
                  <CheckCircle className="h-4 w-4 text-green-600" />
                ) : (
                  <XCircle className="h-4 w-4 text-gray-400" />
                )}
                <span className="text-sm font-medium">
                  {imageAnalysis.faceDetected ? 'Yes' : 'No'}
                </span>
              </div>
            </div>

            {/* Image Quality */}
            <div className="space-y-2">
              <div className="flex items-center justify-between">
                <span className="text-sm font-medium">Image Quality</span>
                <span className={`text-sm font-medium ${getScoreColor(imageAnalysis.imageQuality)}`}>
                  {imageAnalysis.imageQuality}%
                </span>
              </div>
              <Progress 
                value={imageAnalysis.imageQuality} 
                className="h-2"
              />
            </div>

            {/* Manipulation Detection */}
            <div className="space-y-2">
              <div className="flex items-center justify-between">
                <span className="text-sm font-medium">Manipulation Risk</span>
                <span className={`text-sm font-medium ${imageAnalysis.manipulation > 30 ? 'text-red-600' : 'text-green-600'}`}>
                  {imageAnalysis.manipulation}%
                </span>
              </div>
              <Progress 
                value={imageAnalysis.manipulation} 
                className="h-2"
              />
            </div>

            {/* Metadata Verification */}
            <div className="space-y-2">
              <span className="text-sm font-medium">Metadata Verification</span>
              <div className="space-y-1">
                <div className="flex items-center justify-between text-xs">
                  <span className="text-muted-foreground">Original Source</span>
                  {imageAnalysis.metadata.originalSource ? (
                    <CheckCircle className="h-3 w-3 text-green-600" />
                  ) : (
                    <XCircle className="h-3 w-3 text-red-600" />
                  )}
                </div>
                <div className="flex items-center justify-between text-xs">
                  <span className="text-muted-foreground">Date Consistency</span>
                  {imageAnalysis.metadata.dateConsistency ? (
                    <CheckCircle className="h-3 w-3 text-green-600" />
                  ) : (
                    <XCircle className="h-3 w-3 text-red-600" />
                  )}
                </div>
                <div className="flex items-center justify-between text-xs">
                  <span className="text-muted-foreground">Location Consistency</span>
                  {imageAnalysis.metadata.locationConsistency ? (
                    <CheckCircle className="h-3 w-3 text-green-600" />
                  ) : (
                    <XCircle className="h-3 w-3 text-red-600" />
                  )}
                </div>
              </div>
            </div>

            {/* Similar Images */}
            <div className="flex items-center justify-between text-sm">
              <span className="text-muted-foreground">Similar Images Found</span>
              <span className="font-medium">{imageAnalysis.similarImages}</span>
            </div>
          </CardContent>
        </Card>
      )}

      {/* Profile Metrics Card */}
      {profileMetrics && (
        <Card>
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              <User className="h-5 w-5" />
              Profile Metrics
            </CardTitle>
          </CardHeader>
          <CardContent className="space-y-4">
            {/* Account Age */}
            <div className="flex items-center justify-between">
              <span className="text-sm font-medium">Account Age</span>
              <span className="text-sm font-medium">
                {Math.floor(profileMetrics.accountAge / 365)} years
              </span>
            </div>

            {/* Followers Ratio */}
            <div className="space-y-2">
              <div className="flex items-center justify-between">
                <span className="text-sm font-medium">Followers Ratio</span>
                <span className={`text-sm font-medium ${
                  profileMetrics.followersToFollowing > 1 ? 'text-green-600' : 
                  profileMetrics.followersToFollowing > 0.5 ? 'text-yellow-600' : 'text-red-600'
                }`}>
                  {profileMetrics.followersToFollowing.toFixed(2)}
                </span>
              </div>
            </div>

            {/* Engagement Rate */}
            <div className="space-y-2">
              <div className="flex items-center justify-between">
                <span className="text-sm font-medium">Engagement Rate</span>
                <span className={`text-sm font-medium ${getScoreColor(profileMetrics.engagement.rate)}`}>
                  {profileMetrics.engagement.rate}%
                </span>
              </div>
              <Progress 
                value={profileMetrics.engagement.rate} 
                className="h-2"
              />
            </div>

            {/* Activity Pattern */}
            <div className="flex items-center justify-between">
              <span className="text-sm font-medium">Activity Pattern</span>
              <Badge 
                variant={
                  profileMetrics.activityPattern === 'consistent' ? 'default' :
                  profileMetrics.activityPattern === 'suspicious' ? 'destructive' : 'secondary'
                }
                className="text-xs"
              >
                {profileMetrics.activityPattern}
              </Badge>
            </div>

            {/* Verification */}
            <div className="space-y-2">
              <span className="text-sm font-medium">Verification Status</span>
              <div className="space-y-1">
                <div className="flex items-center justify-between text-xs">
                  <span className="text-muted-foreground">Email</span>
                  {profileMetrics.verification.email ? (
                    <CheckCircle className="h-3 w-3 text-green-600" />
                  ) : (
                    <XCircle className="h-3 w-3 text-red-600" />
                  )}
                </div>
                <div className="flex items-center justify-between text-xs">
                  <span className="text-muted-foreground">Phone</span>
                  {profileMetrics.verification.phone ? (
                    <CheckCircle className="h-3 w-3 text-green-600" />
                  ) : (
                    <XCircle className="h-3 w-3 text-red-600" />
                  )}
                </div>
                <div className="flex items-center justify-between text-xs">
                  <span className="text-muted-foreground">Identity</span>
                  {profileMetrics.verification.identity ? (
                    <CheckCircle className="h-3 w-3 text-green-600" />
                  ) : (
                    <XCircle className="h-3 w-3 text-red-600" />
                  )}
                </div>
              </div>
            </div>

            {/* Risk Factors */}
            {profileMetrics.riskFactors.length > 0 && (
              <div className="space-y-2">
                <span className="text-sm font-medium text-red-600">Risk Factors</span>
                <div className="space-y-1">
                  {profileMetrics.riskFactors.slice(0, 2).map((factor, index) => (
                    <div key={index} className="text-xs text-red-600 flex items-center gap-1">
                      <AlertCircle className="h-3 w-3" />
                      {factor}
                    </div>
                  ))}
                </div>
              </div>
            )}
          </CardContent>
        </Card>
      )}
    </div>
  )
}