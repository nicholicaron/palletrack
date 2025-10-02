import { Grid, SimpleGrid } from "@chakra-ui/react"
import { FiAward, FiTrendingDown, FiTrendingUp, FiUsers } from "react-icons/fi"

import MetricCard from "../MetricCard"
import StatChart from "../StatChart"

const TeamPerformance = () => {
  // TODO: Fetch from API - GET /api/workforce/scans-per-operator
  const scansPerOperator = [
    { name: "John D.", scans: 287 },
    { name: "Sarah M.", scans: 265 },
    { name: "Mike T.", scans: 248 },
    { name: "Lisa K.", scans: 232 },
    { name: "Tom R.", scans: 215 },
    { name: "Emma W.", scans: 198 },
  ]

  // TODO: Fetch from API - GET /api/workforce/error-rates
  const errorRatesByOperator = [
    { name: "John D.", errorRate: 2 },
    { name: "Sarah M.", errorRate: 3 },
    { name: "Mike T.", errorRate: 5 },
    { name: "Lisa K.", errorRate: 8 },
    { name: "Tom R.", errorRate: 12 },
    { name: "Emma W.", errorRate: 15 },
  ]

  // TODO: Fetch from API - GET /api/workforce/training-needs
  const trainingNeeds = [
    { operator: "Emma W.", reason: "High manual override rate (15%)" },
    { operator: "Tom R.", reason: "Frequent document placement errors" },
    { operator: "Lisa K.", reason: "Slow processing speed" },
  ]

  // TODO: Fetch from API - GET /api/workforce/best-practices
  const bestPractices = {
    topPerformer: "John D.",
    avgScansPerHour: 35.9,
    errorRate: 2,
    efficiency: 98,
  }

  return (
    <Grid gap={6}>
      <SimpleGrid columns={{ base: 1, md: 2, lg: 4 }} gap={4}>
        <MetricCard
          title="Top Performer"
          value={bestPractices.topPerformer}
          subtitle={`${bestPractices.avgScansPerHour} scans/hour`}
          icon={<FiAward />}
          colorScheme="green"
        />

        <MetricCard
          title="Best Error Rate"
          value={`${bestPractices.errorRate}%`}
          subtitle={bestPractices.topPerformer}
          icon={<FiTrendingDown />}
          colorScheme="green"
        />

        <MetricCard
          title="Training Needed"
          value={trainingNeeds.length}
          subtitle="Operators requiring additional training"
          icon={<FiUsers />}
          colorScheme="yellow"
        />

        <MetricCard
          title="Team Efficiency"
          value={`${bestPractices.efficiency}%`}
          subtitle="Overall team performance"
          icon={<FiTrendingUp />}
          colorScheme="green"
        />
      </SimpleGrid>

      <Grid templateColumns={{ base: "1fr", lg: "1fr 1fr" }} gap={6}>
        <StatChart
          title="Scans Processed by Operator"
          subtitle="Today's productivity"
          type="bar"
          data={scansPerOperator}
          dataKey="scans"
          xAxisKey="name"
          height={300}
          colorScheme="#3182ce"
          hideLegend={true}
        />

        <StatChart
          title="Error Rates by Operator"
          subtitle="Manual override percentage"
          type="bar"
          data={errorRatesByOperator}
          dataKey="errorRate"
          xAxisKey="name"
          height={300}
          colorScheme="#ed8936"
          hideLegend={true}
        />
      </Grid>
    </Grid>
  )
}

export default TeamPerformance
