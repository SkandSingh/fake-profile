'use client'

import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card'
import { Badge } from '@/components/ui/badge'
import { 
  BarChart, 
  Bar, 
  XAxis, 
  YAxis, 
  CartesianGrid, 
  Tooltip, 
  ResponsiveContainer,
  Cell
} from 'recharts'
import { BarChart3, TrendingUp, AlertTriangle } from 'lucide-react'

interface AnalysisData {
  category: string
  score: number
  weight: number
  weightedScore: number
  status: 'high' | 'medium' | 'low'
  details: string[]
}

interface TrustFactorsChartProps {
  data?: AnalysisData[]
  isLoading?: boolean
  overallScore?: number
}

const defaultData: AnalysisData[] = [
  {
    category: 'Text Analysis',
    score: 85,
    weight: 30,
    weightedScore: 25.5,
    status: 'high',
    details: ['Positive sentiment', 'Low toxicity', 'High authenticity']
  },
  {
    category: 'Image Analysis', 
    score: 72,
    weight: 25,
    weightedScore: 18,
    status: 'medium',
    details: ['Good quality', 'Low manipulation risk', 'Metadata verified']
  },
  {
    category: 'Profile Metrics',
    score: 68,
    weight: 25,
    weightedScore: 17,
    status: 'medium',
    details: ['Established account', 'Normal activity', 'Basic verification']
  },
  {
    category: 'Behavioral Patterns',
    score: 78,
    weight: 20,
    weightedScore: 15.6,
    status: 'high',
    details: ['Consistent posting', 'Human-like engagement', 'No bot indicators']
  }
]

export function TrustFactorsChart({ 
  data = defaultData, 
  isLoading = false, 
  overallScore = 76 
}: TrustFactorsChartProps) {
  
  const getBarColor = (status: string) => {
    switch (status) {
      case 'high': return '#22c55e' // green-500
      case 'medium': return '#eab308' // yellow-500
      case 'low': return '#ef4444' // red-500
      default: return '#6b7280' // gray-500
    }
  }

  const getStatusIcon = (status: string) => {
    switch (status) {
      case 'high': return <TrendingUp className="h-4 w-4 text-green-600" />
      case 'medium': return <BarChart3 className="h-4 w-4 text-yellow-600" />
      case 'low': return <AlertTriangle className="h-4 w-4 text-red-600" />
      default: return null
    }
  }

  const getStatusBadgeVariant = (status: string) => {
    switch (status) {
      case 'high': return 'default' as const
      case 'medium': return 'secondary' as const
      case 'low': return 'destructive' as const
      default: return 'outline' as const
    }
  }

  const CustomTooltip = ({ active, payload, label }: any) => {
    if (active && payload && payload.length) {
      const data = payload[0].payload
      return (
        <div className="bg-white dark:bg-gray-800 border rounded-lg shadow-lg p-3">
          <p className="font-medium">{label}</p>
          <div className="space-y-1 text-sm">
            <p className="text-blue-600">
              Score: <span className="font-medium">{data.score}%</span>
            </p>
            <p className="text-purple-600">
              Weight: <span className="font-medium">{data.weight}%</span>
            </p>
            <p className="text-green-600">
              Weighted Score: <span className="font-medium">{data.weightedScore.toFixed(1)}</span>
            </p>
            <div className="pt-1 border-t">
              <p className="text-xs text-muted-foreground">Key factors:</p>
              {data.details.map((detail: string, index: number) => (
                <p key={index} className="text-xs">• {detail}</p>
              ))}
            </div>
          </div>
        </div>
      )
    }
    return null
  }

  if (isLoading) {
    return (
      <Card className="w-full">
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <BarChart3 className="h-5 w-5" />
            Trust Factors Analysis
          </CardTitle>
        </CardHeader>
        <CardContent>
          <div className="space-y-4">
            <div className="h-64 bg-gray-100 dark:bg-gray-800 rounded animate-pulse" />
            <div className="grid grid-cols-2 gap-4">
              {[1, 2, 3, 4].map((i) => (
                <div key={i} className="space-y-2">
                  <div className="h-4 bg-gray-200 rounded animate-pulse" />
                  <div className="h-3 bg-gray-100 rounded animate-pulse w-3/4" />
                </div>
              ))}
            </div>
          </div>
        </CardContent>
      </Card>
    )
  }

  return (
    <Card className="w-full">
      <CardHeader>
        <CardTitle className="flex items-center gap-2">
          <BarChart3 className="h-5 w-5" />
          Trust Factors Analysis
        </CardTitle>
        <div className="flex items-center gap-2">
          <span className="text-sm text-muted-foreground">Overall Trust Score:</span>
          <Badge variant="outline" className="font-medium">
            {overallScore}%
          </Badge>
        </div>
      </CardHeader>
      <CardContent className="space-y-6">
        {/* Bar Chart */}
        <div className="h-64">
          <ResponsiveContainer width="100%" height="100%">
            <BarChart
              data={data}
              margin={{
                top: 20,
                right: 30,
                left: 20,
                bottom: 5,
              }}
            >
              <CartesianGrid strokeDasharray="3 3" className="opacity-30" />
              <XAxis 
                dataKey="category" 
                angle={-45}
                textAnchor="end"
                height={80}
                interval={0}
                fontSize={12}
              />
              <YAxis 
                label={{ value: 'Score (%)', angle: -90, position: 'insideLeft' }}
                fontSize={12}
              />
              <Tooltip content={<CustomTooltip />} />
              <Bar dataKey="score" radius={[4, 4, 0, 0]}>
                {data.map((entry, index) => (
                  <Cell key={`cell-${index}`} fill={getBarColor(entry.status)} />
                ))}
              </Bar>
            </BarChart>
          </ResponsiveContainer>
        </div>

        {/* Factor Breakdown */}
        <div className="space-y-4">
          <h4 className="font-medium text-sm">Factor Breakdown</h4>
          <div className="grid gap-3">
            {data.map((factor, index) => (
              <div key={index} className="flex items-center justify-between p-3 bg-muted/50 rounded-lg">
                <div className="flex items-center gap-3">
                  {getStatusIcon(factor.status)}
                  <div>
                    <div className="font-medium text-sm">{factor.category}</div>
                    <div className="text-xs text-muted-foreground">
                      Weight: {factor.weight}% • Contribution: {factor.weightedScore.toFixed(1)}
                    </div>
                  </div>
                </div>
                <div className="flex items-center gap-2">
                  <Badge variant={getStatusBadgeVariant(factor.status)} className="text-xs">
                    {factor.status}
                  </Badge>
                  <span className="font-medium text-sm">{factor.score}%</span>
                </div>
              </div>
            ))}
          </div>
        </div>

        {/* Calculation Summary */}
        <div className="bg-blue-50 dark:bg-blue-950/30 p-4 rounded-lg border">
          <h4 className="font-medium text-sm mb-2 flex items-center gap-2">
            <BarChart3 className="h-4 w-4" />
            Score Calculation
          </h4>
          <div className="text-xs text-muted-foreground space-y-1">
            <p>Overall trust score is calculated using weighted contributions:</p>
            <div className="grid grid-cols-2 gap-2 mt-2">
              {data.map((factor, index) => (
                <div key={index} className="flex justify-between">
                  <span>{factor.category}:</span>
                  <span className="font-mono">
                    {factor.score}% × {factor.weight}% = {factor.weightedScore.toFixed(1)}
                  </span>
                </div>
              ))}
            </div>
            <div className="border-t pt-2 mt-2 font-medium">
              <div className="flex justify-between">
                <span>Total:</span>
                <span className="font-mono">
                  {data.reduce((sum, factor) => sum + factor.weightedScore, 0).toFixed(1)}%
                </span>
              </div>
            </div>
          </div>
        </div>
      </CardContent>
    </Card>
  )
}