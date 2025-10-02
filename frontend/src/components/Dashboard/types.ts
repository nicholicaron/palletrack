// Dashboard Data Types
// These types define the structure of data fetched from the backend API

// Primary Metrics Types
export interface ThroughputMetrics {
  today: number
  yesterday: number
  weeklyAverage: number
  percentChange: number
  currentRate: {
    palletsPerHour: number
    percentChange: number
  }
  timeSaved: {
    hoursToday: number
    percentChange: number
  }
  peakActivity: Array<{
    name: string
    pallets: number
  }>
}

export interface AccuracyMetrics {
  documentCompleteness: {
    percentage: number
    palletsComplete: number
    palletsMissing: number
    trend: number
  }
  ocrConfidence: {
    averageScore: number
    trend: number
  }
  missingDocs: {
    count: number
    types: string[]
  }
  accuracyTrend: Array<{
    name: string
    score: number
  }>
}

export interface ShipmentStatusMetrics {
  readyToShip: number
  pendingReview: number
  flaggedIssues: number
  inLoadingBay: number
}

// Operational Insights Types
export interface DocumentationComplianceMetrics {
  documentTypeBreakdown: Array<{
    name: string
    value: number
  }>
  missingDocTypes: {
    mostCommon: string
    count: number
  }
  carrierRejections: {
    highestRate: string
    rate: number
  }
  complianceTrend: Array<{
    name: string
    completeness: number
  }>
}

export interface SpeedEfficiencyMetrics {
  avgScanTime: {
    minutes: number
    percentChange: number
  }
  bottleneck: {
    stage: string
    avgTime: number
  }
  topOperator: {
    name: string
    efficiency: number
  }
  timeComparison: {
    manual: number
    automated: number
    savings: number
  }
  stageBreakdown: Array<{
    name: string
    seconds: number
  }>
  operatorPerformance: Array<{
    name: string
    score: number
  }>
}

export interface QualityControlMetrics {
  readabilityByLocation: Array<{
    name: string
    score: number
  }>
  poorLightingZone: {
    location: string
    score: number
  }
  commonErrors: Array<{
    name: string
    count: number
  }>
  manualVerification: {
    count: number
    percentage: number
  }
  environmentalImpact: {
    weatherAffected: number
    condition: string
  }
}

// Carrier & Routing Types
export interface CarrierPerformanceMetrics {
  carrierBreakdown: Array<{
    name: string
    value: number
  }>
  carrierCompliance: Array<{
    name: string
    compliance: number
  }>
  topCarrier: {
    name: string
    compliance: number
  }
  labelIssues: {
    carrier: string
    count: number
  }
  specialHandling: {
    hazmat: number
    tempControlled: number
  }
}

export interface DestinationAnalyticsMetrics {
  topDestinations: Array<{
    name: string
    value: number
  }>
  problemLocations: Array<{
    name: string
    issues: number
  }>
  internationalRatio: {
    domestic: number
    international: number
    percentage: number
  }
  repeatCustomers: {
    count: number
    percentage: number
  }
}

// Risk & Exceptions Types
export interface Alert {
  id: number
  severity: "critical" | "warning" | "info"
  message: string
  count: number
}

export interface AlertSummary {
  critical: number
  warning: number
  info: number
}

export interface ComplianceMetrics {
  hazmat: {
    total: number
    compliant: number
    percentage: number
  }
  tempControlled: {
    total: number
    compliant: number
    percentage: number
  }
  restrictedGoods: {
    total: number
    compliant: number
    percentage: number
  }
  upcomingDeadlines: Array<{
    id: number
    item: string
    daysRemaining: number
  }>
  complianceTrends: Array<{
    name: string
    hazmat: number
    temp: number
    restricted: number
  }>
}

// Historical Trends Types
export interface PerformanceMetrics {
  shipmentVolume: Array<{
    name: string
    pallets: number
  }>
  accuracyImprovement: Array<{
    name: string
    accuracy: number
  }>
  seasonalData: Array<{
    name: string
    pallets: number
  }>
  costSavings: Array<{
    name: string
    manual: number
    automated: number
  }>
  trendSummary: {
    volumeTrend: number
    accuracyTrend: number
    peakMonth: string
    totalSavings: number
  }
}

export interface ProblemPatternsMetrics {
  shiftPerformance: Array<{
    name: string
    accuracy: number
    errors: number
  }>
  problematicSKUs: Array<{
    name: string
    issues: number
  }>
  vendorIssues: Array<{
    name: string
    issues: number
  }>
  rejectionCauses: Array<{
    name: string
    value: number
  }>
  patternSummary: {
    worstShift: string
    shiftAccuracyGap: number
    topProblemSKU: string
    worstVendor: string
  }
}

// Workforce Management Types
export interface TeamPerformanceMetrics {
  scansPerOperator: Array<{
    name: string
    scans: number
  }>
  errorRatesByOperator: Array<{
    name: string
    errorRate: number
  }>
  trainingNeeds: Array<{
    operator: string
    reason: string
  }>
  bestPractices: {
    topPerformer: string
    avgScansPerHour: number
    errorRate: number
    efficiency: number
  }
}

export interface ResourceAllocationMetrics {
  busyTimes: Array<{
    name: string
    scans: number
    staffNeeded: number
  }>
  projectedWorkload: Array<{
    name: string
    pallets: number
  }>
  equipmentUtilization: {
    activeHours: number
    totalHours: number
    utilizationRate: number
    idleTime: number
  }
  staffingRecommendations: {
    peakTime: string
    additionalStaff: number
    projectedTomorrow: number
    recommendedStaff: number
  }
}

// Financial Impact Types
export interface CostBenefitMetrics {
  laborSavings: {
    hoursPerDay: number
    hoursPerWeek: number
    hoursPerMonth: number
    costPerHour: number
    dailySavings: number
    weeklySavings: number
    monthlySavings: number
  }
  rejectionReduction: {
    beforeAutomation: number
    afterAutomation: number
    reductionPercentage: number
    avgCostPerRejection: number
    monthlySavings: number
  }
  detentionFees: {
    previousMonth: number
    currentMonth: number
    reduction: number
    savings: number
  }
  customerSatisfaction: {
    score: number
    improvement: number
    delayedShipments: number
    onTimePercentage: number
  }
  monthlyComparison: Array<{
    name: string
    manual: number
    automated: number
  }>
  savingsBreakdown: Array<{
    name: string
    value: number
  }>
}

export interface ProblemCostsMetrics {
  rejectedShipmentCosts: {
    count: number
    totalCost: number
    avgCostPerRejection: number
    trend: number
  }
  expeditedShipping: {
    count: number
    totalCost: number
    avgCost: number
    trend: number
  }
  customerPenalties: {
    count: number
    totalCost: number
    avgPenalty: number
    trend: number
  }
  monthlyProblemCosts: Array<{
    name: string
    rejections: number
    expedited: number
    penalties: number
  }>
  costByIssueType: Array<{
    name: string
    cost: number
  }>
}

// Actionable Insights Types
export interface Recommendation {
  id: number
  type: "critical" | "improvement" | "optimization" | "info"
  title: string
  description: string
  impact: "high" | "medium" | "low"
  category: string
}

export interface DocumentTemplateStats {
  totalFormats: number
  problematicFormats: number
  standardizationOpportunity: number
}
