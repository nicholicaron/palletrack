import { Grid, SimpleGrid } from "@chakra-ui/react"
import { FiAlertCircle, FiFileText, FiTrendingDown, FiTruck } from "react-icons/fi"

import MetricCard from "../MetricCard"
import StatChart from "../StatChart"

const DocumentationCompliance = () => {
  // TODO: Fetch from API - GET /api/metrics/compliance/document-types
  const documentTypeBreakdown = [
    { name: "BOL", value: 98 },
    { name: "Packing List", value: 92 },
    { name: "Shipping Label", value: 100 },
    { name: "Customs Docs", value: 85 },
    { name: "Hazmat Cert", value: 88 },
  ]

  // TODO: Fetch from API - GET /api/metrics/compliance/missing-docs
  const missingDocTypes = {
    mostCommon: "Packing List",
    count: 27,
  }

  // TODO: Fetch from API - GET /api/metrics/compliance/carrier-rejections
  const carrierRejections = {
    highestRate: "Regional LTL",
    rate: 8.5,
  }

  // TODO: Fetch from API - GET /api/metrics/compliance/trends
  const complianceTrend = [
    { name: "Week 1", completeness: 88 },
    { name: "Week 2", completeness: 90 },
    { name: "Week 3", completeness: 92 },
    { name: "Week 4", completeness: 94 },
  ]

  return (
    <Grid gap={6}>
      <SimpleGrid columns={{ base: 1, md: 2, lg: 4 }} gap={4}>
        <MetricCard
          title="Most Complete Doc Type"
          value="Shipping Label"
          subtitle="100% present on all shipments"
          icon={<FiFileText />}
          colorScheme="green"
        />

        <MetricCard
          title="Most Missing Doc"
          value={missingDocTypes.mostCommon}
          subtitle={`Missing on ${missingDocTypes.count} shipments`}
          icon={<FiAlertCircle />}
          colorScheme="yellow"
        />

        <MetricCard
          title="Highest Rejection Rate"
          value={carrierRejections.highestRate + "%"}
          subtitle={carrierRejections.highestRate}
          icon={<FiTruck />}
          colorScheme="red"
        />

        <MetricCard
          title="Compliance Trend"
          value="Improving"
          subtitle="Up 6% this month"
          icon={<FiTrendingDown />}
          colorScheme="green"
          trend={{
            value: 6,
            label: "this month",
            isPositive: true,
          }}
        />
      </SimpleGrid>

      <Grid templateColumns={{ base: "1fr", lg: "1fr 1fr" }} gap={6}>
        <StatChart
          title="Document Type Completeness"
          subtitle="Percentage of shipments with each document type"
          type="bar"
          data={documentTypeBreakdown}
          dataKey="value"
          xAxisKey="name"
          height={300}
          hideLegend={true}
        />

        <StatChart
          title="Compliance Improvement Trend"
          subtitle="Document completeness over time"
          type="area"
          data={complianceTrend}
          dataKey="completeness"
          xAxisKey="name"
          height={300}
          colorScheme="#38b2ac"
        />
      </Grid>
    </Grid>
  )
}

export default DocumentationCompliance
