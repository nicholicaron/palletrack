import { Grid, SimpleGrid } from "@chakra-ui/react"
import { FiActivity, FiClock, FiTrendingUp, FiZap } from "react-icons/fi"

import MetricCard from "../MetricCard"
import StatChart from "../StatChart"

const ResourceAllocation = () => {
  // TODO: Fetch from API - GET /api/workforce/busy-times
  const busyTimesData = [
    { name: "6am", scans: 15, staffNeeded: 2 },
    { name: "8am", scans: 45, staffNeeded: 3 },
    { name: "10am", scans: 68, staffNeeded: 4 },
    { name: "12pm", scans: 52, staffNeeded: 3 },
    { name: "2pm", scans: 89, staffNeeded: 5 },
    { name: "4pm", scans: 67, staffNeeded: 4 },
    { name: "6pm", scans: 12, staffNeeded: 2 },
  ]

  // TODO: Fetch from API - GET /api/workforce/projected-workload
  const projectedWorkload = [
    { name: "Mon", pallets: 340 },
    { name: "Tue", pallets: 360 },
    { name: "Wed", pallets: 380 },
    { name: "Thu", pallets: 420 },
    { name: "Fri", pallets: 450 },
  ]

  // TODO: Fetch from API - GET /api/workforce/equipment-utilization
  const equipmentUtilization = {
    activeHours: 8.5,
    totalHours: 10,
    utilizationRate: 85,
    idleTime: 1.5,
  }

  // TODO: Fetch from API - GET /api/workforce/recommendations
  const staffingRecommendations = {
    peakTime: "2-4 PM",
    additionalStaff: 2,
    projectedTomorrow: 420,
    recommendedStaff: 6,
  }

  return (
    <Grid gap={6}>
      <SimpleGrid columns={{ base: 1, md: 2, lg: 4 }} gap={4}>
        <MetricCard
          title="Peak Staffing Time"
          value={staffingRecommendations.peakTime}
          subtitle={`Need ${staffingRecommendations.additionalStaff} additional operators`}
          icon={<FiClock />}
          colorScheme="blue"
        />

        <MetricCard
          title="Equipment Utilization"
          value={`${equipmentUtilization.utilizationRate}%`}
          subtitle={`${equipmentUtilization.activeHours}h active, ${equipmentUtilization.idleTime}h idle`}
          icon={<FiActivity />}
          colorScheme="green"
        />

        <MetricCard
          title="Tomorrow's Forecast"
          value={`${staffingRecommendations.projectedTomorrow} pallets`}
          subtitle={`Recommend ${staffingRecommendations.recommendedStaff} operators`}
          icon={<FiTrendingUp />}
          colorScheme="blue"
        />

        <MetricCard
          title="System Efficiency"
          value="High"
          subtitle="Minimal idle time detected"
          icon={<FiZap />}
          colorScheme="green"
        />
      </SimpleGrid>

      <Grid templateColumns={{ base: "1fr", lg: "1fr 1fr" }} gap={6}>
        <StatChart
          title="Hourly Activity & Staffing Needs"
          subtitle="Scanning volume and recommended staff by hour"
          type="bar"
          data={busyTimesData}
          dataKey="scans"
          xAxisKey="name"
          height={300}
          colorScheme="#3182ce"
          hideLegend={true}
        >
          {/* TODO: Add second bar series for staffNeeded */}
        </StatChart>

        <StatChart
          title="Projected Weekly Workload"
          subtitle="Expected pallets to process this week"
          type="line"
          data={projectedWorkload}
          dataKey="pallets"
          xAxisKey="name"
          height={300}
          colorScheme="#805ad5"
        />
      </Grid>
    </Grid>
  )
}

export default ResourceAllocation
