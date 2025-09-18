'use client'

import { useState } from 'react'
import { Button } from '@/components/ui/button'
import { 
  DropdownMenu, 
  DropdownMenuContent, 
  DropdownMenuItem, 
  DropdownMenuTrigger 
} from '@/components/ui/dropdown-menu'
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card'
import { Badge } from '@/components/ui/badge'
import { 
  Download, 
  FileJson, 
  FileText, 
  Image as ImageIcon, 
  Share, 
  ChevronDown,
  CheckCircle
} from 'lucide-react'

interface ExportData {
  id: string
  timestamp: Date
  inputType: 'url' | 'file'
  inputValue: string
  trustScore: number
  textAnalysis?: any
  imageAnalysis?: any
  profileMetrics?: any
}

interface ResultExporterProps {
  data: ExportData
  onExport?: (format: 'json' | 'pdf' | 'csv') => void
}

export function ResultExporter({ data, onExport }: ResultExporterProps) {
  const [isExporting, setIsExporting] = useState<string | null>(null)

  const exportAsJSON = () => {
    setIsExporting('json')
    
    const exportData = {
      analysis_report: {
        id: data.id,
        timestamp: data.timestamp.toISOString(),
        input: {
          type: data.inputType,
          value: data.inputValue
        },
        overall_score: data.trustScore,
        detailed_analysis: {
          text_analysis: data.textAnalysis,
          image_analysis: data.imageAnalysis,
          profile_metrics: data.profileMetrics
        },
        metadata: {
          export_timestamp: new Date().toISOString(),
          format_version: "1.0",
          generated_by: "Profile Trust Analyzer"
        }
      }
    }
    
    const dataStr = JSON.stringify(exportData, null, 2)
    const dataBlob = new Blob([dataStr], { type: 'application/json' })
    const url = URL.createObjectURL(dataBlob)
    const link = document.createElement('a')
    link.href = url
    link.download = `profile-analysis-report-${data.id}.json`
    link.click()
    URL.revokeObjectURL(url)
    
    setTimeout(() => {
      setIsExporting(null)
      onExport?.('json')
    }, 1000)
  }

  const exportAsCSV = () => {
    setIsExporting('csv')
    
    const csvData = [
      // Headers
      ['Field', 'Value'],
      ['Analysis ID', data.id],
      ['Timestamp', data.timestamp.toISOString()],
      ['Input Type', data.inputType],
      ['Input Value', data.inputValue],
      ['Overall Trust Score', `${data.trustScore}%`],
      [''],
      
      // Text Analysis
      ...(data.textAnalysis ? [
        ['--- Text Analysis ---', ''],
        ['Sentiment', data.textAnalysis.sentiment],
        ['Sentiment Score', `${data.textAnalysis.sentimentScore}%`],
        ['Toxicity', `${data.textAnalysis.toxicity}%`],
        ['Authenticity', `${data.textAnalysis.authenticity}%`],
        ['Language', data.textAnalysis.languageDetected],
        ['Keywords', data.textAnalysis.keywords.join(', ')],
        ['']
      ] : []),
      
      // Image Analysis
      ...(data.imageAnalysis ? [
        ['--- Image Analysis ---', ''],
        ['Face Detected', data.imageAnalysis.faceDetected ? 'Yes' : 'No'],
        ['Image Quality', `${data.imageAnalysis.imageQuality}%`],
        ['Manipulation Risk', `${data.imageAnalysis.manipulation}%`],
        ['Original Source', data.imageAnalysis.metadata.originalSource ? 'Yes' : 'No'],
        ['Date Consistency', data.imageAnalysis.metadata.dateConsistency ? 'Yes' : 'No'],
        ['Similar Images Found', data.imageAnalysis.similarImages.toString()],
        ['']
      ] : []),
      
      // Profile Metrics
      ...(data.profileMetrics ? [
        ['--- Profile Metrics ---', ''],
        ['Account Age (days)', data.profileMetrics.accountAge.toString()],
        ['Followers Ratio', data.profileMetrics.followersToFollowing.toFixed(2)],
        ['Engagement Rate', `${data.profileMetrics.engagement.rate}%`],
        ['Activity Pattern', data.profileMetrics.activityPattern],
        ['Email Verified', data.profileMetrics.verification.email ? 'Yes' : 'No'],
        ['Phone Verified', data.profileMetrics.verification.phone ? 'Yes' : 'No'],
        ['Identity Verified', data.profileMetrics.verification.identity ? 'Yes' : 'No'],
        ['Risk Factors', data.profileMetrics.riskFactors.join('; ')]
      ] : [])
    ]
    
    const csvContent = csvData.map(row => 
      row.map(field => `"${field.toString().replace(/"/g, '""')}"`).join(',')
    ).join('\n')
    
    const dataBlob = new Blob([csvContent], { type: 'text/csv' })
    const url = URL.createObjectURL(dataBlob)
    const link = document.createElement('a')
    link.href = url
    link.download = `profile-analysis-report-${data.id}.csv`
    link.click()
    URL.revokeObjectURL(url)
    
    setTimeout(() => {
      setIsExporting(null)
      onExport?.('csv')
    }, 1000)
  }

  const exportAsPDF = async () => {
    setIsExporting('pdf')
    
    // Create a comprehensive HTML report for PDF export
    const htmlContent = `
      <!DOCTYPE html>
      <html>
      <head>
        <meta charset="utf-8">
        <title>Profile Trust Analysis Report</title>
        <style>
          body { font-family: Arial, sans-serif; margin: 40px; color: #333; }
          .header { text-align: center; margin-bottom: 40px; border-bottom: 2px solid #3b82f6; padding-bottom: 20px; }
          .section { margin-bottom: 30px; }
          .section h2 { color: #3b82f6; border-bottom: 1px solid #e5e7eb; padding-bottom: 10px; }
          .metric { display: flex; justify-content: space-between; margin: 10px 0; padding: 10px; background: #f9fafb; }
          .score { font-weight: bold; font-size: 24px; color: ${data.trustScore >= 80 ? '#16a34a' : data.trustScore >= 60 ? '#eab308' : '#ef4444'}; }
          .high { color: #16a34a; } .medium { color: #eab308; } .low { color: #ef4444; }
          .footer { margin-top: 40px; padding-top: 20px; border-top: 1px solid #e5e7eb; font-size: 12px; color: #6b7280; }
        </style>
      </head>
      <body>
        <div class="header">
          <h1>Profile Trust Analysis Report</h1>
          <p><strong>Analysis ID:</strong> ${data.id}</p>
          <p><strong>Generated:</strong> ${new Date().toLocaleString()}</p>
          <p><strong>Input:</strong> ${data.inputValue} (${data.inputType})</p>
        </div>
        
        <div class="section">
          <h2>Overall Trust Score</h2>
          <div style="text-align: center;">
            <div class="score">${data.trustScore}%</div>
            <p>${data.trustScore >= 80 ? 'Highly Trustworthy' : data.trustScore >= 60 ? 'Moderately Trustworthy' : data.trustScore >= 40 ? 'Questionable' : 'High Risk'}</p>
          </div>
        </div>
        
        ${data.textAnalysis ? `
        <div class="section">
          <h2>Text Analysis</h2>
          <div class="metric"><span>Sentiment:</span><span class="${data.textAnalysis.sentiment === 'positive' ? 'high' : data.textAnalysis.sentiment === 'negative' ? 'low' : 'medium'}">${data.textAnalysis.sentiment} (${data.textAnalysis.sentimentScore}%)</span></div>
          <div class="metric"><span>Authenticity:</span><span>${data.textAnalysis.authenticity}%</span></div>
          <div class="metric"><span>Toxicity:</span><span>${data.textAnalysis.toxicity}%</span></div>
          <div class="metric"><span>Language:</span><span>${data.textAnalysis.languageDetected}</span></div>
          <div class="metric"><span>Keywords:</span><span>${data.textAnalysis.keywords.join(', ')}</span></div>
        </div>
        ` : ''}
        
        ${data.imageAnalysis ? `
        <div class="section">
          <h2>Image Analysis</h2>
          <div class="metric"><span>Face Detected:</span><span>${data.imageAnalysis.faceDetected ? 'Yes' : 'No'}</span></div>
          <div class="metric"><span>Image Quality:</span><span>${data.imageAnalysis.imageQuality}%</span></div>
          <div class="metric"><span>Manipulation Risk:</span><span>${data.imageAnalysis.manipulation}%</span></div>
          <div class="metric"><span>Original Source:</span><span>${data.imageAnalysis.metadata.originalSource ? 'Verified' : 'Not Verified'}</span></div>
          <div class="metric"><span>Similar Images:</span><span>${data.imageAnalysis.similarImages} found</span></div>
        </div>
        ` : ''}
        
        ${data.profileMetrics ? `
        <div class="section">
          <h2>Profile Metrics</h2>
          <div class="metric"><span>Account Age:</span><span>${Math.floor(data.profileMetrics.accountAge / 365)} years</span></div>
          <div class="metric"><span>Followers Ratio:</span><span>${data.profileMetrics.followersToFollowing.toFixed(2)}</span></div>
          <div class="metric"><span>Engagement Rate:</span><span>${data.profileMetrics.engagement.rate}%</span></div>
          <div class="metric"><span>Activity Pattern:</span><span>${data.profileMetrics.activityPattern}</span></div>
          <div class="metric"><span>Verification Status:</span><span>Email: ${data.profileMetrics.verification.email ? '✓' : '✗'}, Phone: ${data.profileMetrics.verification.phone ? '✓' : '✗'}, Identity: ${data.profileMetrics.verification.identity ? '✓' : '✗'}</span></div>
          ${data.profileMetrics.riskFactors.length > 0 ? `<div class="metric"><span>Risk Factors:</span><span style="color: #ef4444;">${data.profileMetrics.riskFactors.join(', ')}</span></div>` : ''}
        </div>
        ` : ''}
        
        <div class="footer">
          <p>This report was generated by Profile Trust Analyzer. The analysis is based on available data and should be used as guidance only.</p>
          <p><strong>Disclaimer:</strong> Results may vary based on data quality and analysis algorithms. Always verify information through multiple sources.</p>
        </div>
      </body>
      </html>
    `
    
    // Create a new window and print
    const printWindow = window.open('', '_blank')
    if (printWindow) {
      printWindow.document.write(htmlContent)
      printWindow.document.close()
      
      // Wait for content to load, then trigger print dialog
      setTimeout(() => {
        printWindow.print()
        printWindow.close()
        setIsExporting(null)
        onExport?.('pdf')
      }, 1000)
    }
  }

  return (
    <div className="flex items-center gap-2">
      <DropdownMenu>
        <DropdownMenuTrigger asChild>
          <Button variant="outline" disabled={!!isExporting}>
            {isExporting ? (
              <>
                <div className="animate-spin rounded-full h-4 w-4 border-b-2 border-blue-600 mr-2" />
                Exporting...
              </>
            ) : (
              <>
                <Download className="h-4 w-4 mr-2" />
                Export Report
                <ChevronDown className="h-4 w-4 ml-2" />
              </>
            )}
          </Button>
        </DropdownMenuTrigger>
        <DropdownMenuContent align="end" className="w-48">
          <DropdownMenuItem onClick={exportAsJSON} className="cursor-pointer">
            <FileJson className="h-4 w-4 mr-2" />
            Export as JSON
            <Badge variant="secondary" className="ml-auto text-xs">
              Data
            </Badge>
          </DropdownMenuItem>
          <DropdownMenuItem onClick={exportAsCSV} className="cursor-pointer">
            <FileText className="h-4 w-4 mr-2" />
            Export as CSV
            <Badge variant="secondary" className="ml-auto text-xs">
              Table
            </Badge>
          </DropdownMenuItem>
          <DropdownMenuItem onClick={exportAsPDF} className="cursor-pointer">
            <FileText className="h-4 w-4 mr-2" />
            Export as PDF
            <Badge variant="secondary" className="ml-auto text-xs">
              Report
            </Badge>
          </DropdownMenuItem>
        </DropdownMenuContent>
      </DropdownMenu>
      
      <Button variant="outline" size="sm">
        <Share className="h-4 w-4 mr-2" />
        Share
      </Button>
    </div>
  )
}

// Export summary component for quick overview
export function ExportSummary({ totalExports }: { totalExports: number }) {
  return (
    <Card className="w-full">
      <CardHeader className="pb-3">
        <CardTitle className="flex items-center gap-2 text-lg">
          <Download className="h-5 w-5" />
          Export Analytics
        </CardTitle>
      </CardHeader>
      <CardContent>
        <div className="grid grid-cols-3 gap-4 text-center">
          <div>
            <div className="text-2xl font-bold text-blue-600">{totalExports}</div>
            <div className="text-sm text-muted-foreground">Total Exports</div>
          </div>
          <div>
            <div className="text-2xl font-bold text-green-600">JSON</div>
            <div className="text-sm text-muted-foreground">Most Popular</div>
          </div>
          <div>
            <div className="flex items-center justify-center">
              <CheckCircle className="h-6 w-6 text-green-600" />
            </div>
            <div className="text-sm text-muted-foreground">Available</div>
          </div>
        </div>
      </CardContent>
    </Card>
  )
}