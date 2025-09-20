'use client'

import { useEffect, useState } from 'react'
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card'
import { Badge } from '@/components/ui/badge'
import { Shield, ShieldCheck, ShieldX, AlertTriangle } from 'lucide-react'

interface TrustScoreProps {
  score: number // 0-100
  isLoading?: boolean
  breakdown?: {
    textAnalysis: number
    imageAnalysis: number
    profileMetrics: number
  }
}

export function TrustScore({ score, isLoading = false, breakdown }: TrustScoreProps) {
  const [animatedScore, setAnimatedScore] = useState(0)
  
  useEffect(() => {
    if (!isLoading && score > 0) {
      const timer = setTimeout(() => {
        const increment = score / 50 // Animate over ~1 second
        let current = 0
        const animate = () => {
          current += increment
          if (current < score) {
            setAnimatedScore(Math.min(current, score))
            requestAnimationFrame(animate)
          } else {
            setAnimatedScore(score)
          }
        }
        animate()
      }, 500)
      
      return () => clearTimeout(timer)
    }
  }, [score, isLoading])

  const getScoreColor = (score: number) => {
    if (score >= 80) return 'text-green-600'
    if (score >= 60) return 'text-yellow-600'
    if (score >= 40) return 'text-orange-600'
    return 'text-red-600'
  }

  const getScoreBgColor = (score: number) => {
    if (score >= 80) return 'from-green-500 to-green-600'
    if (score >= 60) return 'from-yellow-500 to-yellow-600'
    if (score >= 40) return 'from-orange-500 to-orange-600'
    return 'from-red-500 to-red-600'
  }

  const getScoreLabel = (score: number) => {
    if (score >= 80) return 'Highly Trustworthy'
    if (score >= 60) return 'Moderately Trustworthy'
    if (score >= 40) return 'Questionable'
    return 'High Risk'
  }

  const getScoreIcon = (score: number) => {
    if (score >= 80) return <ShieldCheck className="h-6 w-6 text-green-600" />
    if (score >= 60) return <Shield className="h-6 w-6 text-yellow-600" />
    if (score >= 40) return <AlertTriangle className="h-6 w-6 text-orange-600" />
    return <ShieldX className="h-6 w-6 text-red-600" />
  }

  const getBadgeVariant = (score: number) => {
    if (score >= 80) return 'success' as const
    if (score >= 60) return 'warning' as const
    if (score >= 40) return 'warning' as const
    return 'danger' as const
  }

  const circumference = 2 * Math.PI * 90 // radius = 90
  const strokeDasharray = circumference
  const strokeDashoffset = circumference - (animatedScore / 100) * circumference

  return (
    <Card className="w-full">
      <CardHeader className="text-center">
        <CardTitle className="flex items-center justify-center gap-2">
          {getScoreIcon(score)}
          Trust Score
        </CardTitle>
      </CardHeader>
      <CardContent className="space-y-6">
        {/* Circular Progress */}
        <div className="relative flex items-center justify-center">
          <svg
            width="200"
            height="200"
            viewBox="0 0 200 200"
            className="transform -rotate-90"
          >
            {/* Background circle */}
            <circle
              cx="100"
              cy="100"
              r="90"
              fill="none"
              stroke="currentColor"
              strokeWidth="8"
              className="text-gray-200"
            />
            
            {/* Progress circle */}
            {!isLoading && (
              <circle
                cx="100"
                cy="100"
                r="90"
                fill="none"
                strokeWidth="8"
                strokeLinecap="round"
                strokeDasharray={strokeDasharray}
                strokeDashoffset={strokeDashoffset}
                className={`transition-all duration-1000 ease-out bg-gradient-to-r ${getScoreBgColor(score)}`}
                style={{
                  stroke: `url(#gradient-${Math.floor(score / 20)})`
                }}
              />
            )}
            
            {/* Gradient definitions */}
            <defs>
              <linearGradient id="gradient-0" x1="0%" y1="0%" x2="100%" y2="0%">
                <stop offset="0%" stopColor="#ef4444" />
                <stop offset="100%" stopColor="#dc2626" />
              </linearGradient>
              <linearGradient id="gradient-1" x1="0%" y1="0%" x2="100%" y2="0%">
                <stop offset="0%" stopColor="#f97316" />
                <stop offset="100%" stopColor="#ea580c" />
              </linearGradient>
              <linearGradient id="gradient-2" x1="0%" y1="0%" x2="100%" y2="0%">
                <stop offset="0%" stopColor="#eab308" />
                <stop offset="100%" stopColor="#ca8a04" />
              </linearGradient>
              <linearGradient id="gradient-3" x1="0%" y1="0%" x2="100%" y2="0%">
                <stop offset="0%" stopColor="#eab308" />
                <stop offset="100%" stopColor="#ca8a04" />
              </linearGradient>
              <linearGradient id="gradient-4" x1="0%" y1="0%" x2="100%" y2="0%">
                <stop offset="0%" stopColor="#22c55e" />
                <stop offset="100%" stopColor="#16a34a" />
              </linearGradient>
            </defs>
          </svg>
          
          {/* Score display */}
          <div className="absolute inset-0 flex flex-col items-center justify-center">
            {isLoading ? (
              <div className="space-y-2 text-center">
                <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-blue-600 mx-auto" />
                <p className="text-sm text-muted-foreground">Analyzing...</p>
              </div>
            ) : (
              <>
                <div className={`text-4xl font-bold ${getScoreColor(score)}`}>
                  {Math.round(animatedScore)}
                </div>
                <div className="text-lg font-medium text-muted-foreground">/ 100</div>
              </>
            )}
          </div>
        </div>

        {/* Score Label */}
        {!isLoading && (
          <div className="text-center space-y-3">
            <Badge variant={getBadgeVariant(score)} className="text-sm px-4 py-1">
              {getScoreLabel(score)}
            </Badge>
            
            {/* Score Breakdown */}
            {breakdown && (
              <div className="space-y-3 pt-4 border-t">
                <h4 className="font-medium text-sm">Analysis Breakdown</h4>
                <div className="space-y-2">
                  <div className="flex items-center justify-between text-sm">
                    <span className="text-muted-foreground">Text Analysis</span>
                    <span className={`font-medium ${getScoreColor(breakdown.textAnalysis)}`}>
                      {breakdown.textAnalysis}%
                    </span>
                  </div>
                  <div className="flex items-center justify-between text-sm">
                    <span className="text-muted-foreground">Image Analysis</span>
                    <span className={`font-medium ${getScoreColor(breakdown.imageAnalysis)}`}>
                      {breakdown.imageAnalysis}%
                    </span>
                  </div>
                  <div className="flex items-center justify-between text-sm">
                    <span className="text-muted-foreground">Profile Metrics</span>
                    <span className={`font-medium ${getScoreColor(breakdown.profileMetrics)}`}>
                      {breakdown.profileMetrics}%
                    </span>
                  </div>
                </div>
              </div>
            )}
          </div>
        )}
      </CardContent>
    </Card>
  )
}