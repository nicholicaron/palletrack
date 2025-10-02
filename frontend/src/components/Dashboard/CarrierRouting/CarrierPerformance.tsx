import { Grid, SimpleGrid } from "@chakra-ui/react"
import { FiAlertCircle, FiCheckCircle, FiPackage, FiTruck } from "react-icons/fi"

import MetricCard from "../MetricCard"
import StatChart from "../StatChart"

const CarrierPerformance = () => {
  // TODO: Fetch from API - GET /api/metrics/carriers/breakdown
  const carrierBreakdown = [
    { name: "UPS", value: 145 },
    { name: "FedEx", value: 98 },
    { name: "Regional LTL", value: 67 },
    { name: "DHL", value: 32 },
  ]

  // TODO: Fetch from API - GET /api/metrics/carriers/compliance
  const carrierCompliance = [
    { name: "UPS", compliance: 98 },
    { name: "FedEx", compliance: 96 },
    { name: "Regional LTL", compliance: 88 },
    { name: "DHL", compliance: 94 },
  ]

  // TODO: Fetch from API - GET /api/metrics/carriers/top-carrier
  const topCarrier = {
    name: "UPS",
    compliance: 98,
  }

  // TODO: Fetch from API - GET /api/metrics/carriers/label-issues
  const labelIssues = {
    carrier: "Regional LTL",
    count: 8,
  }

  // TODO: Fetch from API - GET /api/metrics/carriers/special-handling
  const specialHandling = {
    hazmat: 12,
    tempControlled: 8,
  }

  return (
    <Grid gap={6}>
      <SimpleGrid columns={{ base: 1, md: 2, lg: 4 }} gap={4}>
        <MetricCard
          title="Top Carrier by Volume"
          value={topCarrier.name}
          subtitle="145 shipments today"
          icon={<FiTruck />}
          colorScheme="blue"
        />

        <MetricCard
          title="Best Compliance"
          value={topCarrier.name}
          subtitle={`${topCarrier.compliance}% documentation accuracy`}
          icon={<FiCheckCircle />}
          colorScheme="green"
        />

        <MetricCard
          title="Label Format Issues"
          value={labelIssues.carrier}
          subtitle={`${labelIssues.count} format errors today`}
          icon={<FiAlertCircle />}
          colorScheme="yellow"
        />

        <MetricCard
          title="Special Handling"
          value={specialHandling.hazmat + specialHandling.tempControlled}
          subtitle={`${specialHandling.hazmat} Hazmat, ${specialHandling.tempControlled} Temp-Controlled`}
          icon={<FiPackage />}
          colorScheme="blue"
        />
      </SimpleGrid>

      <Grid templateColumns={{ base: "1fr", lg: "1fr 1fr" }} gap={6}>
        <StatChart
          title="Shipments by Carrier"
          subtitle="Today's distribution"
          type="pie"
          data={carrierBreakdown}
          dataKey="value"
          xAxisKey="name"
          height={300}
          colorScheme="#805ad5"
        />

        <StatChart
          title="Carrier Documentation Compliance"
          subtitle="Percentage of shipments with complete docs"
          type="bar"
          data={carrierCompliance}
          dataKey="compliance"
          xAxisKey="name"
          height={300}
          colorScheme="#3182ce"
          hideLegend={true}
        />
      </Grid>
    </Grid>
  )
}

export default CarrierPerformance
