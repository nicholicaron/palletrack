import { Grid, SimpleGrid } from "@chakra-ui/react"
import { FiCalendar, FiTrendingDown, FiTrendingUp } from "react-icons/fi"

import MetricCard from "../MetricCard"
import StatChart from "../StatChart"

const PerformanceOverTime = () => {
  // TODO: Fetch from API - GET /api/trends/shipment-volume
  const shipmentVolumeData = [
    { name: "Week 1", pallets: 1420 },
    { name: "Week 2", pallets: 1580 },
    { name: "Week 3", pallets: 1690 },
    { name: "Week 4", pallets: 1750 },
    { name: "Week 5", pallets: 1820 },
    { name: "Week 6", pallets: 1780 },
    { name: "Week 7", pallets: 1850 },
    { name: "Week 8", pallets: 1920 },
  ]

  // TODO: Fetch from API - GET /api/trends/accuracy-improvement
  const accuracyImprovementData = [
    { name: "Week 1", accuracy: 88 },
    { name: "Week 2", accuracy: 89 },
    { name: "Week 3", accuracy: 91 },
    { name: "Week 4", accuracy: 92 },
    { name: "Week 5", accuracy: 93 },
    { name: "Week 6", accuracy: 94 },
    { name: "Week 7", accuracy: 95 },
    { name: "Week 8", accuracy: 96 },
  ]

  // TODO: Fetch from API - GET /api/trends/seasonal-patterns
  const seasonalData = [
    { name: "Jan", pallets: 1200 },
    { name: "Feb", pallets: 1100 },
    { name: "Mar", pallets: 1400 },
    { name: "Apr", pallets: 1500 },
    { name: "May", pallets: 1600 },
    { name: "Jun", pallets: 1700 },
    { name: "Jul", pallets: 1650 },
    { name: "Aug", pallets: 1800 },
    { name: "Sep", pallets: 1900 },
    { name: "Oct", pallets: 2100 },
    { name: "Nov", pallets: 2400 },
    { name: "Dec", pallets: 1800 },
  ]

  // TODO: Fetch from API - GET /api/trends/cost-savings
  const costSavingsData = [
    { name: "Month 1", manual: 24000, automated: 8500 },
    { name: "Month 2", manual: 25000, automated: 8200 },
    { name: "Month 3", manual: 26000, automated: 8000 },
    { name: "Month 4", manual: 26500, automated: 7800 },
  ]

  // TODO: Fetch from API - GET /api/trends/summary
  const trendSummary = {
    volumeTrend: 15.2,
    accuracyTrend: 9.1,
    peakMonth: "November",
    totalSavings: 68000,
  }

  return (
    <Grid gap={6}>
      <SimpleGrid columns={{ base: 1, md: 2, lg: 4 }} gap={4}>
        <MetricCard
          title="Volume Trend"
          value={`+${trendSummary.volumeTrend}%`}
          subtitle="8-week growth rate"
          trend={{
            value: trendSummary.volumeTrend,
            label: "growth",
            isPositive: true,
          }}
          icon={<FiTrendingUp />}
          colorScheme="green"
        />

        <MetricCard
          title="Accuracy Improvement"
          value={`+${trendSummary.accuracyTrend}%`}
          subtitle="Since automation started"
          trend={{
            value: trendSummary.accuracyTrend,
            label: "improvement",
            isPositive: true,
          }}
          icon={<FiTrendingUp />}
          colorScheme="green"
        />

        <MetricCard
          title="Peak Month"
          value={trendSummary.peakMonth}
          subtitle="Historically highest volume"
          icon={<FiCalendar />}
          colorScheme="blue"
        />

        <MetricCard
          title="Total Savings YTD"
          value={`$${(trendSummary.totalSavings / 1000).toFixed(0)}k`}
          subtitle="vs manual processing costs"
          icon={<FiTrendingDown />}
          colorScheme="green"
        />
      </SimpleGrid>

      <Grid templateColumns={{ base: "1fr", lg: "1fr 1fr" }} gap={6}>
        <StatChart
          title="Weekly Shipment Volume Trend"
          subtitle="Pallets processed per week"
          type="area"
          data={shipmentVolumeData}
          dataKey="pallets"
          xAxisKey="name"
          height={300}
          colorScheme="#3182ce"
        />

        <StatChart
          title="Documentation Accuracy Improvement"
          subtitle="Accuracy percentage over time"
          type="line"
          data={accuracyImprovementData}
          dataKey="accuracy"
          xAxisKey="name"
          height={300}
          colorScheme="#48bb78"
        />
      </Grid>

      <Grid templateColumns={{ base: "1fr", lg: "1fr 1fr" }} gap={6}>
        <StatChart
          title="Seasonal Shipping Patterns"
          subtitle="Monthly volume distribution"
          type="bar"
          data={seasonalData}
          dataKey="pallets"
          xAxisKey="name"
          colorScheme="#805ad5"
          useSingleColor={true}
          hideLegend={true}
        />

        <StatChart
          title="Cost Savings Trend"
          subtitle="Manual vs Automated processing costs"
          type="line"
          data={costSavingsData}
          dataKey="manual"
          xAxisKey="name"
          height={300}
          colorScheme="#e53e3e"
        >
          {/* TODO: Add second line series for automated costs */}
        </StatChart>
      </Grid>
    </Grid>
  )
}

export default PerformanceOverTime
