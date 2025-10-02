import { Grid, SimpleGrid } from "@chakra-ui/react"
import { FiGlobe, FiMapPin, FiRefreshCw, FiAlertCircle } from "react-icons/fi"

import MetricCard from "../MetricCard"
import StatChart from "../StatChart"

const DestinationAnalytics = () => {
  // TODO: Fetch from API - GET /api/metrics/destinations/top-destinations
  const topDestinations = [
    { name: "Los Angeles, CA", value: 45 },
    { name: "Chicago, IL", value: 38 },
    { name: "Houston, TX", value: 32 },
    { name: "New York, NY", value: 28 },
    { name: "Atlanta, GA", value: 22 },
  ]

  // TODO: Fetch from API - GET /api/metrics/destinations/problem-locations
  const problemLocations = [
    { name: "Miami, FL", issues: 8 },
    { name: "Seattle, WA", issues: 6 },
    { name: "Boston, MA", issues: 4 },
  ]

  // TODO: Fetch from API - GET /api/metrics/destinations/international-ratio
  const internationalRatio = {
    domestic: 312,
    international: 30,
    percentage: 8.8,
  }

  // TODO: Fetch from API - GET /api/metrics/destinations/repeat-customers
  const repeatCustomers = {
    count: 23,
    percentage: 67,
  }

  return (
    <Grid gap={6}>
      <SimpleGrid columns={{ base: 1, md: 2, lg: 4 }} gap={4}>
        <MetricCard
          title="Top Destination"
          value="Los Angeles, CA"
          subtitle="45 shipments today"
          icon={<FiMapPin />}
          colorScheme="blue"
        />

        <MetricCard
          title="International Shipments"
          value={internationalRatio.international}
          subtitle={`${internationalRatio.percentage}% of total volume`}
          icon={<FiGlobe />}
          colorScheme="blue"
        />

        <MetricCard
          title="Repeat Destinations"
          value={repeatCustomers.count}
          subtitle={`${repeatCustomers.percentage}% of shipments`}
          icon={<FiRefreshCw />}
          colorScheme="green"
        />

        <MetricCard
          title="Problem Locations"
          value={problemLocations.length}
          subtitle="Destinations with chronic doc issues"
          icon={<FiAlertCircle />}
          colorScheme="yellow"
        />
      </SimpleGrid>

      <Grid templateColumns={{ base: "1fr", lg: "1fr 1fr" }} gap={6}>
        <StatChart
          title="Top Shipping Destinations"
          subtitle="This week's most frequent destinations"
          type="bar"
          data={topDestinations}
          dataKey="value"
          xAxisKey="name"
          height={300}
          colorScheme="#38b2ac"
          hideLegend={true}
        />

        <StatChart
          title="Locations with Documentation Issues"
          subtitle="Destinations requiring extra attention"
          type="bar"
          data={problemLocations}
          dataKey="issues"
          xAxisKey="name"
          height={300}
          colorScheme="#ed8936"
          hideLegend={true}
        />
      </Grid>
    </Grid>
  )
}

export default DestinationAnalytics
