import { Grid, SimpleGrid } from "@chakra-ui/react"
import { FiAlertTriangle, FiMapPin, FiRefreshCw, FiSun } from "react-icons/fi"

import MetricCard from "../MetricCard"
import StatChart from "../StatChart"

const QualityControl = () => {
  // TODO: Fetch from API - GET /api/metrics/quality/readability-by-location
  const readabilityByLocation = [
    { name: "Dock 1", score: 98 },
    { name: "Dock 2", score: 95 },
    { name: "Dock 3", score: 72 },
    { name: "Dock 4", score: 91 },
    { name: "Warehouse", score: 88 },
  ]

  // TODO: Fetch from API - GET /api/metrics/quality/poor-lighting-zone
  const poorLightingZone = {
    location: "Dock 3",
    score: 72,
  }

  // TODO: Fetch from API - GET /api/metrics/quality/common-errors
  const commonErrors = [
    { name: "Blurry Text", count: 23 },
    { name: "Poor Contrast", count: 18 },
    { name: "Damaged Label", count: 12 },
    { name: "Partial Doc", count: 8 },
  ]

  // TODO: Fetch from API - GET /api/metrics/quality/manual-verification
  const manualVerification = {
    count: 15,
    percentage: 4.4,
  }

  // TODO: Fetch from API - GET /api/metrics/quality/environmental-impact
  const environmentalImpact = {
    weatherAffected: 6,
    condition: "Rain",
  }

  return (
    <Grid gap={6}>
      <SimpleGrid columns={{ base: 1, md: 2, lg: 4 }} gap={4}>
        <MetricCard
          title="Best Scan Quality"
          value="Dock 1"
          subtitle="98% average readability score"
          icon={<FiMapPin />}
          colorScheme="green"
        />

        <MetricCard
          title="Poor Lighting Zone"
          value={poorLightingZone.location}
          subtitle={`${poorLightingZone.score}% readability - needs improvement`}
          icon={<FiSun />}
          colorScheme="red"
        />

        <MetricCard
          title="Manual Verification"
          value={manualVerification.count}
          subtitle={`${manualVerification.percentage}% of today's scans`}
          icon={<FiRefreshCw />}
          colorScheme="yellow"
        />

        <MetricCard
          title="Weather Impact"
          value={environmentalImpact.weatherAffected}
          subtitle={`Scans affected by ${environmentalImpact.condition}`}
          icon={<FiAlertTriangle />}
          colorScheme="yellow"
        />
      </SimpleGrid>

      <Grid templateColumns={{ base: "1fr", lg: "1fr 1fr" }} gap={6}>
        <StatChart
          title="Readability by Location"
          subtitle="Average OCR confidence score by scanning location"
          type="bar"
          data={readabilityByLocation}
          dataKey="score"
          xAxisKey="name"
          height={300}
          hideLegend={true}
        />

        <StatChart
          title="Common OCR Errors"
          subtitle="Types of errors requiring manual review"
          type="bar"
          data={commonErrors}
          dataKey="count"
          xAxisKey="name"
          height={300}
          hideLegend={true}
        />
      </Grid>
    </Grid>
  )
}

export default QualityControl
