import { Grid, SimpleGrid } from "@chakra-ui/react"
import { FiAlertCircle, FiBox, FiClock, FiUsers } from "react-icons/fi"

import MetricCard from "../MetricCard"
import StatChart from "../StatChart"

const ProblemPatterns = () => {
  // TODO: Fetch from API - GET /api/patterns/shift-performance
  const shiftPerformanceData = [
    { name: "Day Shift", accuracy: 96, errors: 12 },
    { name: "Night Shift", accuracy: 89, errors: 28 },
  ]

  // TODO: Fetch from API - GET /api/patterns/problematic-skus
  const problematicSKUs = [
    { name: "SKU-8472", issues: 23 },
    { name: "SKU-9201", issues: 18 },
    { name: "SKU-7651", issues: 15 },
    { name: "SKU-3398", issues: 12 },
    { name: "SKU-4529", issues: 10 },
  ]

  // TODO: Fetch from API - GET /api/patterns/vendor-analysis
  const vendorIssues = [
    { name: "Acme Corp", issues: 34 },
    { name: "Global Supplies", issues: 28 },
    { name: "Tech Distributors", issues: 22 },
    { name: "Parts Unlimited", issues: 18 },
  ]

  // TODO: Fetch from API - GET /api/patterns/rejection-causes
  const rejectionCauses = [
    { name: "Missing BOL", value: 45 },
    { name: "Damaged Label", value: 32 },
    { name: "Wrong Address", value: 28 },
    { name: "Incomplete Customs", value: 18 },
    { name: "Missing Signature", value: 15 },
  ]

  // TODO: Fetch from API - GET /api/patterns/summary
  const patternSummary = {
    worstShift: "Night Shift",
    shiftAccuracyGap: 7,
    topProblemSKU: "SKU-8472",
    worstVendor: "Acme Corp",
  }

  return (
    <Grid gap={6}>
      <SimpleGrid columns={{ base: 1, md: 2, lg: 4 }} gap={4}>
        <MetricCard
          title="Shift Performance Gap"
          value={`${patternSummary.shiftAccuracyGap}%`}
          subtitle={`${patternSummary.worstShift} needs improvement`}
          icon={<FiClock />}
          colorScheme="yellow"
        />

        <MetricCard
          title="Most Problematic SKU"
          value={patternSummary.topProblemSKU}
          subtitle="23 documentation issues this month"
          icon={<FiBox />}
          colorScheme="red"
        />

        <MetricCard
          title="Worst Vendor Compliance"
          value={patternSummary.worstVendor}
          subtitle="34 incomplete shipments"
          icon={<FiUsers />}
          colorScheme="red"
        />

        <MetricCard
          title="Top Rejection Cause"
          value="Missing BOL"
          subtitle="45 rejections this month"
          icon={<FiAlertCircle />}
          colorScheme="red"
        />
      </SimpleGrid>

      <Grid templateColumns={{ base: "1fr", lg: "1fr 1fr" }} gap={6}>
        <StatChart
          title="Shift Performance Comparison"
          subtitle="Accuracy percentage by shift"
          type="bar"
          data={shiftPerformanceData}
          dataKey="accuracy"
          xAxisKey="name"
          height={300}
          colorScheme="#ed8936"
          hideLegend={true}
        />

        <StatChart
          title="Root Causes of Rejected Shipments"
          subtitle="Breakdown of rejection reasons"
          type="pie"
          data={rejectionCauses}
          dataKey="value"
          xAxisKey="name"
          height={300}
          colorScheme="#e53e3e"
        />
      </Grid>

      <Grid templateColumns={{ base: "1fr", lg: "1fr 1fr" }} gap={6}>
        <StatChart
          title="Problematic SKUs"
          subtitle="Products with most documentation issues"
          type="bar"
          data={problematicSKUs}
          dataKey="issues"
          xAxisKey="name"
          height={300}
          colorScheme="#dd6b20"
          hideLegend={true}
        />

        <StatChart
          title="Vendor Compliance Issues"
          subtitle="Suppliers with poor documentation"
          type="bar"
          data={vendorIssues}
          dataKey="issues"
          xAxisKey="name"
          height={300}
          colorScheme="#c53030"
          hideLegend={true}
        />
      </Grid>
    </Grid>
  )
}

export default ProblemPatterns
