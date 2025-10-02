import { Grid, SimpleGrid } from "@chakra-ui/react"
import { FiActivity, FiClock, FiTrendingUp, FiZap } from "react-icons/fi"

import MetricCard from "../MetricCard"
import StatChart from "../StatChart"

const ThroughputPanel = () => {
  // TODO: Fetch from API - GET /api/metrics/throughput/daily
  const dailyThroughput = {
    today: 342,
    yesterday: 318,
    weeklyAverage: 295,
    percentChange: 7.5,
  }

  // TODO: Fetch from API - GET /api/metrics/throughput/current-rate
  const currentRate = {
    palletsPerHour: 28,
    percentChange: 12.0,
  }

  // TODO: Fetch from API - GET /api/metrics/time-saved
  const timeSaved = {
    hoursToday: 4.2,
    percentChange: 5.0,
  }

  // TODO: Fetch from API - GET /api/metrics/peak-activity
  const peakActivityData = [
    { name: "6am", pallets: 12 },
    { name: "8am", pallets: 45 },
    { name: "10am", pallets: 68 },
    { name: "12pm", pallets: 52 },
    { name: "2pm", pallets: 89 },
    { name: "4pm", pallets: 67 },
    { name: "6pm", pallets: 9 },
  ]

  return (
    <Grid gap={6}>
      <SimpleGrid columns={{ base: 1, md: 2, lg: 4 }} gap={4}>
        <MetricCard
          title="Pallets Scanned Today"
          value={dailyThroughput.today}
          subtitle={`Yesterday: ${dailyThroughput.yesterday} | Avg: ${dailyThroughput.weeklyAverage}`}
          trend={{
            value: dailyThroughput.percentChange,
            label: "vs yesterday",
            isPositive: true,
          }}
          icon={<FiActivity />}
          colorScheme="blue"
        />

        <MetricCard
          title="Current Scanning Rate"
          value={`${currentRate.palletsPerHour}/hr`}
          subtitle="Real-time processing speed"
          trend={{
            value: currentRate.percentChange,
            label: "vs avg",
            isPositive: true,
          }}
          icon={<FiZap />}
          colorScheme="green"
        />

        <MetricCard
          title="Time Saved Today"
          value={`${timeSaved.hoursToday}h`}
          subtitle="vs manual data entry"
          trend={{
            value: timeSaved.percentChange,
            label: "vs yesterday",
            isPositive: true,
          }}
          icon={<FiClock />}
          colorScheme="green"
        />

        <MetricCard
          title="Peak Activity"
          value="2-4 PM"
          subtitle="Most shipments processed"
          icon={<FiTrendingUp />}
          colorScheme="blue"
        />
      </SimpleGrid>

      <StatChart
        title="Peak Activity Hours"
        subtitle="Pallets scanned by time of day"
        type="bar"
        data={peakActivityData}
        dataKey="pallets"
        xAxisKey="name"
        height={250}
        colorScheme="#3182ce"
        useSingleColor={true}
        hideLegend={true}
      />
    </Grid>
  )
}

export default ThroughputPanel
