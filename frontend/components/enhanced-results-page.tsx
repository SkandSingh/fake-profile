'use client'

import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card'
import { Badge } from '@/components/ui/badge'
import { Progress } from '@/components/ui/progress'
import { 
  Shield, 
  ShieldCheck, 
  ShieldX, 
  AlertTriangle,
  FileText,
  Image,
  User,
  CheckCircle,
  XCircle,
  MinusCircle,
  TrendingUp,
  TrendingDown,
  Eye,
  Heart,
  MessageCircle,
  Share
} from 'lucide-react'

interface EnhancedResultsPageProps {
  result: {
    trustScore: number
    textScore: number
    imageScore: number
    metricsScore: number
    explanation: string[]
    breakdown: {
      textAnalysis: any
      imageAnalysis: any
      profileMetrics: any
    }
    processingTime: number
    analysisId: string
  }
}

export function EnhancedResultsPage({ result }: EnhancedResultsPageProps) {
  // Generate detailed red flags from the analysis
  const generateRedFlags = () => {
    const flags: { type: 'high' | 'medium' | 'low', message: string }[] = []
    
    const { textAnalysis, imageAnalysis, profileMetrics } = result.breakdown
    
    // Text analysis red flags
    if (textAnalysis.sentimentScore < 40) {
      flags.push({ type: 'high', message: 'Highly negative sentiment detected' })
    }
    if (textAnalysis.toxicity > 70) {
      flags.push({ type: 'high', message: 'High toxicity levels in content' })
    }
    if (textAnalysis.authenticity < 50) {
      flags.push({ type: 'medium', message: 'Low text authenticity score' })
    }
    if (textAnalysis.coherence?.coherence_score < 0.6) {
      flags.push({ type: 'medium', message: 'Low coherence in text content' })
    }
    
    // Image analysis red flags
    if (imageAnalysis.manipulation > 60) {
      flags.push({ type: 'high', message: 'High probability of image manipulation' })
    }
    if (imageAnalysis.imageQuality < 40) {
      flags.push({ type: 'medium', message: 'Poor image quality detected' })
    }
    if (!imageAnalysis.metadata.originalSource) {
      flags.push({ type: 'medium', message: 'Image may not be original source' })
    }
    
    // Profile metrics red flags
    if (profileMetrics.accountAge < 30) {
      flags.push({ type: 'high', message: 'Very new account (suspicious timing)' })
    }
    if (profileMetrics.followersToFollowing > 10) {
      flags.push({ type: 'high', message: 'Suspicious follower-to-following ratio' })
    }
    if (profileMetrics.activityPattern === 'suspicious') {
      flags.push({ type: 'high', message: 'Suspicious activity patterns detected' })
    }
    if (profileMetrics.engagement.rate < 1) {
      flags.push({ type: 'medium', message: 'Low engagement rate for follower count' })
    }
    if (profileMetrics.riskFactors && profileMetrics.riskFactors.length > 0) {
      profileMetrics.riskFactors.forEach((factor: string) => {
        flags.push({ type: 'medium', message: factor })
      })
    }
    
    return flags
  }

  const redFlags = generateRedFlags()
  
  // Calculate model contributions for bar chart
  const modelContributions = [
    {
      name: 'Text Analysis',
      score: result.textScore,
      contribution: (result.textScore * 0.35), // 35% weight
      color: '#3b82f6' // blue
    },
    {
      name: 'Image Analysis', 
      score: result.imageScore,
      contribution: (result.imageScore * 0.25), // 25% weight
      color: '#10b981' // emerald
    },
    {
      name: 'Profile Metrics',
      score: result.metricsScore,
      contribution: (result.metricsScore * 0.40), // 40% weight
      color: '#8b5cf6' // violet
    }
  ]

  return (
    <div className="space-y-6 md:space-y-8">
      {/* Main Trust Score - Circular Meter */}
      <div className="grid grid-cols-1 xl:grid-cols-2 gap-6 md:gap-8">
        <div className="xl:col-span-1">
          <CircularTrustMeter score={result.trustScore} />
        </div>
        
        {/* Quick Summary */}
        <div className="xl:col-span-1">
          <Card>
            <CardHeader>
              <CardTitle className="flex items-center gap-2">
                <Eye className="h-5 w-5" />
                Analysis Summary
              </CardTitle>
            </CardHeader>
            <CardContent className="space-y-4">
              <div className="grid grid-cols-2 gap-4 text-center">
                <div>
                  <div className="text-xl md:text-2xl font-bold text-blue-600">{result.textScore}</div>
                  <div className="text-xs md:text-sm text-muted-foreground">Text Score</div>
                </div>
                <div>
                  <div className="text-xl md:text-2xl font-bold text-emerald-600">{result.imageScore}</div>
                  <div className="text-xs md:text-sm text-muted-foreground">Image Score</div>
                </div>
              </div>
              <div className="text-center">
                <div className="text-xl md:text-2xl font-bold text-violet-600">{result.metricsScore}</div>
                <div className="text-xs md:text-sm text-muted-foreground">Metrics Score</div>
              </div>
              
              <div className="pt-4 border-t">
                <div className="text-xs md:text-sm text-muted-foreground">Processing Time</div>
                <div className="font-medium">{result.processingTime}ms</div>
              </div>
            </CardContent>
          </Card>
        </div>
      </div>

      {/* Individual Model Cards */}
      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-4 md:gap-6">
        <ModelScoreCard
          title="Text Analysis"
          score={result.textScore}
          icon={<FileText className="h-6 w-6" />}
          data={result.breakdown.textAnalysis}
          color="blue"
        />
        <ModelScoreCard
          title="Image Analysis"
          score={result.imageScore}
          icon={<Image className="h-6 w-6" />}
          data={result.breakdown.imageAnalysis}
          color="emerald"
        />
        <ModelScoreCard
          title="Profile Metrics"
          score={result.metricsScore}
          icon={<User className="h-6 w-6" />}
          data={result.breakdown.profileMetrics}
          color="violet"
        />
      </div>

      {/* Model Contribution Bar Chart */}
      <Card>
        <CardHeader>
          <CardTitle>Model Contribution Analysis</CardTitle>
        </CardHeader>
        <CardContent>
          <ModelContributionChart data={modelContributions} />
        </CardContent>
      </Card>

      {/* Red Flags and Explanations */}
      <div className="grid grid-cols-1 xl:grid-cols-2 gap-4 md:gap-6">
        {/* Red Flags */}
        <Card>
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              <AlertTriangle className="h-5 w-5 text-red-500" />
              Risk Factors ({redFlags.length})
            </CardTitle>
          </CardHeader>
          <CardContent>
            <div className="space-y-3">
              {redFlags.length > 0 ? (
                redFlags.map((flag, index) => (
                  <div key={index} className="flex items-start gap-3 p-3 rounded-lg border">
                    <div className={`mt-0.5 ${
                      flag.type === 'high' ? 'text-red-500' : 
                      flag.type === 'medium' ? 'text-yellow-500' : 'text-blue-500'
                    }`}>
                      {flag.type === 'high' ? <XCircle className="h-4 w-4" /> :
                       flag.type === 'medium' ? <AlertTriangle className="h-4 w-4" /> :
                       <MinusCircle className="h-4 w-4" />}
                    </div>
                    <div className="flex-1">
                      <div className="text-sm font-medium">{flag.message}</div>
                      <Badge 
                        variant={flag.type === 'high' ? 'destructive' : flag.type === 'medium' ? 'secondary' : 'outline'}
                        className="text-xs mt-1"
                      >
                        {flag.type.toUpperCase()} RISK
                      </Badge>
                    </div>
                  </div>
                ))
              ) : (
                <div className="text-center py-8 text-muted-foreground">
                  <CheckCircle className="h-12 w-12 text-green-500 mx-auto mb-3" />
                  <div className="font-medium">No significant risk factors detected</div>
                  <div className="text-sm">Profile appears to be legitimate</div>
                </div>
              )}
            </div>
          </CardContent>
        </Card>

        {/* AI Explanations */}
        <Card>
          <CardHeader>
            <CardTitle>AI Analysis Insights</CardTitle>
          </CardHeader>
          <CardContent>
            <div className="space-y-3">
              {result.explanation.map((explanation, index) => (
                <div key={index} className="flex items-start gap-3 p-3 rounded-lg bg-muted/50">
                  <div className="mt-0.5 text-blue-500">
                    <CheckCircle className="h-4 w-4" />
                  </div>
                  <div className="text-sm">{explanation}</div>
                </div>
              ))}
            </div>
          </CardContent>
        </Card>
      </div>
    </div>
  )
}

// Circular Trust Meter Component
function CircularTrustMeter({ score }: { score: number }) {
  const getScoreColor = (score: number) => {
    if (score >= 80) return { text: 'text-green-600', stroke: '#22c55e' }
    if (score >= 60) return { text: 'text-yellow-600', stroke: '#eab308' }
    if (score >= 40) return { text: 'text-orange-600', stroke: '#f97316' }
    return { text: 'text-red-600', stroke: '#ef4444' }
  }

  const getScoreLabel = (score: number) => {
    if (score >= 80) return 'Highly Trustworthy'
    if (score >= 60) return 'Moderately Trustworthy'
    if (score >= 40) return 'Questionable'
    return 'High Risk'
  }

  const getScoreIcon = (score: number) => {
    if (score >= 80) return <ShieldCheck className="h-8 w-8 text-green-600" />
    if (score >= 60) return <Shield className="h-8 w-8 text-yellow-600" />
    if (score >= 40) return <AlertTriangle className="h-8 w-8 text-orange-600" />
    return <ShieldX className="h-8 w-8 text-red-600" />
  }

  const circumference = 2 * Math.PI * 120
  const strokeDasharray = circumference
  const strokeDashoffset = circumference - (score / 100) * circumference
  const colors = getScoreColor(score)

  return (
    <Card>
      <CardHeader className="text-center">
        <CardTitle className="flex items-center justify-center gap-2 md:gap-3 text-lg md:text-xl">
          {getScoreIcon(score)}
          Overall Trust Score
        </CardTitle>
      </CardHeader>
      <CardContent className="flex flex-col items-center">
        <div className="relative w-64 h-64 md:w-72 md:h-72 lg:w-80 lg:h-80">
          <svg className="w-full h-full" viewBox="0 0 280 280">
            {/* Background circle */}
            <circle
              cx="140"
              cy="140"
              r="120"
              fill="none"
              stroke="rgb(229 231 235)"
              strokeWidth="12"
              className="dark:stroke-gray-700"
            />
            
            {/* Progress circle */}
            <circle
              cx="140"
              cy="140"
              r="120"
              fill="none"
              stroke={colors.stroke}
              strokeWidth="12"
              strokeLinecap="round"
              strokeDasharray={strokeDasharray}
              strokeDashoffset={strokeDashoffset}
              transform="rotate(-90 140 140)"
              className="transition-all duration-1000 ease-out"
            />
          </svg>
          
          {/* Score display */}
          <div className="absolute inset-0 flex flex-col items-center justify-center">
            <div className={`text-4xl md:text-5xl lg:text-6xl font-bold ${colors.text}`}>
              {score}
            </div>
            <div className="text-lg md:text-xl text-muted-foreground">/ 100</div>
            <div className="text-sm md:text-base lg:text-lg font-medium mt-2 text-center px-2">
              {getScoreLabel(score)}
            </div>
          </div>
        </div>
      </CardContent>
    </Card>
  )
}

// Model Score Card Component
function ModelScoreCard({ 
  title, 
  score, 
  icon, 
  data, 
  color 
}: { 
  title: string
  score: number
  icon: React.ReactNode
  data: any
  color: 'blue' | 'emerald' | 'violet'
}) {
  const colorClasses = {
    blue: 'text-blue-600 border-blue-200 bg-blue-50',
    emerald: 'text-emerald-600 border-emerald-200 bg-emerald-50',
    violet: 'text-violet-600 border-violet-200 bg-violet-50'
  }

  const getSubMetrics = () => {
    if (title === 'Text Analysis') {
      return [
        { label: 'Sentiment', value: data.sentimentScore || 0 },
        { label: 'Authenticity', value: data.authenticity || 0 },
        { label: 'Toxicity', value: 100 - (data.toxicity || 0) } // Invert toxicity
      ]
    } else if (title === 'Image Analysis') {
      return [
        { label: 'Quality', value: data.imageQuality || 0 },
        { label: 'Authenticity', value: 100 - (data.manipulation || 0) }, // Invert manipulation
        { label: 'Confidence', value: data.confidence || 0 }
      ]
    } else {
      return [
        { label: 'Account Age', value: Math.min((data.accountAge / 365) * 25, 100) || 0 },
        { label: 'Engagement', value: (data.engagement?.rate || 0) * 10 },
        { label: 'Verification', value: (Object.values(data.verification || {}).filter(Boolean).length / 3) * 100 }
      ]
    }
  }

  return (
    <Card className={`border-2 ${colorClasses[color]}`}>
      <CardHeader className="pb-3">
        <CardTitle className={`flex items-center gap-2 text-lg ${colorClasses[color].split(' ')[0]}`}>
          {icon}
          {title}
        </CardTitle>
      </CardHeader>
      <CardContent className="space-y-4">
        <div className="text-center">
          <div className={`text-4xl font-bold ${colorClasses[color].split(' ')[0]}`}>
            {score}
          </div>
          <div className="text-sm text-muted-foreground">/ 100</div>
        </div>
        
        <div className="space-y-3">
          {getSubMetrics().map((metric, index) => (
            <div key={index} className="space-y-1">
              <div className="flex justify-between text-sm">
                <span className="text-muted-foreground">{metric.label}</span>
                <span className="font-medium">{Math.round(metric.value)}%</span>
              </div>
              <Progress 
                value={metric.value} 
                className="h-2" 
              />
            </div>
          ))}
        </div>
      </CardContent>
    </Card>
  )
}

// Model Contribution Bar Chart
function ModelContributionChart({ data }: { data: Array<{ name: string, score: number, contribution: number, color: string }> }) {
  const maxContribution = Math.max(...data.map(d => d.contribution))
  
  return (
    <div className="space-y-6">
      <div className="text-sm text-muted-foreground text-center">
        Weighted contribution of each model to the final trust score
      </div>
      
      <div className="space-y-4">
        {data.map((item, index) => (
          <div key={index} className="space-y-2">
            <div className="flex justify-between items-center">
              <span className="font-medium">{item.name}</span>
              <div className="flex items-center gap-2">
                <span className="text-sm text-muted-foreground">
                  {item.score}% → {item.contribution.toFixed(1)} pts
                </span>
              </div>
            </div>
            
            <div className="relative">
              <div className="w-full bg-gray-200 rounded-full h-6">
                <div
                  className="h-6 rounded-full flex items-center justify-end pr-3 text-white text-sm font-medium transition-all duration-700 ease-out"
                  style={{
                    width: `${(item.contribution / maxContribution) * 100}%`,
                    backgroundColor: item.color
                  }}
                >
                  {item.contribution.toFixed(1)}
                </div>
              </div>
            </div>
          </div>
        ))}
      </div>
      
      <div className="pt-4 border-t">
        <div className="text-center">
          <div className="text-sm text-muted-foreground">Total Weighted Score</div>
          <div className="text-2xl font-bold">
            {data.reduce((sum, item) => sum + item.contribution, 0).toFixed(1)} / 100
          </div>
        </div>
      </div>
    </div>
  )
}