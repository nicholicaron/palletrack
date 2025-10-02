import { Grid, SimpleGrid } from "@chakra-ui/react"
import { FiClock, FiTrendingUp, FiUsers, FiZap } from "react-icons/fi"

import MetricCard from "../MetricCard"
import StatChart from "../StatChart"

const SpeedEfficiency = () => {
  // TODO: Fetch from API - GET /api/metrics/efficiency/avg-time
  const avgScanTime = {
    minutes: 2.3,
    percentChange: -15.2,
  }

  // TODO: Fetch from API - GET /api/metrics/efficiency/bottlenecks
  const bottleneck = {
    stage: "OCR Processing",
    avgTime: 45,
  }

  // TODO: Fetch from API - GET /api/metrics/efficiency/top-operator
  const topOperator = {
    name: "John D.",
    efficiency: 98,
  }

  // TODO: Fetch from API - GET /api/metrics/efficiency/comparison
  const timeComparison = {
    manual: 8.5,
    automated: 2.3,
    savings: 73,
  }

  // TODO: Fetch from API - GET /api/metrics/efficiency/stage-breakdown
  const stageBreakdown = [
    { name: "Detection", seconds: 15 },
    { name: "OCR", seconds: 45 },
    { name: "Validation", seconds: 30 },
    { name: "Storage", seconds: 8 },
  ]

  // TODO: Fetch from API - GET /api/metrics/efficiency/operator-performance
  const operatorPerformance = [
    { name: "John D.", score: 98 },
    { name: "Sarah M.", score: 95 },
    { name: "Mike T.", score: 92 },
    { name: "Lisa K.", score: 88 },
    { name: "Tom R.", score: 85 },
  ]

  return (
    <Grid gap={6}>
      <SimpleGrid columns={{ base: 1, md: 2, lg: 4 }} gap={4}>
        <MetricCard
          title="Avg Time Per Pallet"
          value={`${avgScanTime.minutes}min`}
          subtitle="From arrival to scan completion"
          trend={{
            value: Math.abs(avgScanTime.percentChange),
            label: "faster vs last week",
            isPositive: true,
          }}
          icon={<FiClock />}
          colorScheme="green"
        />

        <MetricCard
          title="Primary Bottleneck"
          value={bottleneck.stage}
          subtitle={`Avg ${bottleneck.avgTime}s processing time`}
          icon={<FiTrendingUp />}
          colorScheme="yellow"
        />

        <MetricCard
          title="Top Performer"
          value={topOperator.name}
          subtitle={`${topOperator.efficiency}% efficiency rating`}
          icon={<FiUsers />}
          colorScheme="blue"
        />

        <MetricCard
          title="Time Savings"
          value={`${timeComparison.savings}%`}
          subtitle={`Automated (${timeComparison.automated}m) vs Manual (${timeComparison.manual}m)`}
          icon={<FiZap />}
          colorScheme="green"
        />
      </SimpleGrid>

      <Grid templateColumns={{ base: "1fr", lg: "1fr 1fr" }} gap={6}>
        <StatChart
          title="Processing Stage Breakdown"
          subtitle="Average time spent in each stage (seconds)"
          type="bar"
          data={stageBreakdown}
          dataKey="seconds"
          xAxisKey="name"
          height={300}
          hideLegend={true}
        />

        <StatChart
          title="Operator Performance"
          subtitle="Efficiency scores by operator"
          type="bar"
          data={operatorPerformance}
          dataKey="score"
          xAxisKey="name"
          height={300}
          hideLegend={true}
        />
      </Grid>
    </Grid>
  )
}

export default SpeedEfficiency
