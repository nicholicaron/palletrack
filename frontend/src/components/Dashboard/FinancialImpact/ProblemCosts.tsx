import { Grid, SimpleGrid } from "@chakra-ui/react"
import { FiAlertTriangle, FiDollarSign, FiTrendingDown, FiTruck } from "react-icons/fi"

import MetricCard from "../MetricCard"
import StatChart from "../StatChart"

const ProblemCosts = () => {
  // TODO: Fetch from API - GET /api/financial/rejected-shipments
  const rejectedShipmentCosts = {
    count: 8,
    totalCost: 1200,
    avgCostPerRejection: 150,
    trend: -25,
  }

  // TODO: Fetch from API - GET /api/financial/expedited-shipping
  const expeditedShipping = {
    count: 12,
    totalCost: 2200,
    avgCost: 183,
    trend: -40,
  }

  // TODO: Fetch from API - GET /api/financial/customer-penalties
  const customerPenalties = {
    count: 2,
    totalCost: 800,
    avgPenalty: 400,
    trend: -60,
  }

  // TODO: Fetch from API - GET /api/financial/monthly-problem-costs
  const monthlyProblemCosts = [
    { name: "Jan", rejections: 6800, expedited: 8200, penalties: 3200 },
    { name: "Feb", rejections: 5200, expedited: 6500, penalties: 2400 },
    { name: "Mar", rejections: 3400, expedited: 4200, penalties: 1600 },
    { name: "Apr", rejections: 1200, expedited: 2200, penalties: 800 },
  ]

  // TODO: Fetch from API - GET /api/financial/cost-by-issue-type
  const costByIssueType = [
    { name: "Missing Docs", cost: 3500 },
    { name: "Wrong Address", cost: 2800 },
    { name: "Damaged Labels", cost: 1900 },
    { name: "Late Shipments", cost: 1200 },
    { name: "Other", cost: 800 },
  ]

  const totalMonthlyCosts =
    rejectedShipmentCosts.totalCost +
    expeditedShipping.totalCost +
    customerPenalties.totalCost

  const previousMonthCosts = 6800 + 8200 + 3200
  const costReduction = Math.round(
    ((previousMonthCosts - totalMonthlyCosts) / previousMonthCosts) * 100
  )

  return (
    <Grid gap={6}>
      <SimpleGrid columns={{ base: 1, md: 2, lg: 4 }} gap={4}>
        <MetricCard
          title="Rejected Shipment Costs"
          value={`$${rejectedShipmentCosts.totalCost.toLocaleString()}`}
          subtitle={`${rejectedShipmentCosts.count} rejections this month`}
          trend={{
            value: Math.abs(rejectedShipmentCosts.trend),
            label: "vs last month",
            isPositive: true,
          }}
          icon={<FiAlertTriangle />}
          colorScheme="green"
        />

        <MetricCard
          title="Expedited Shipping"
          value={`$${expeditedShipping.totalCost.toLocaleString()}`}
          subtitle={`${expeditedShipping.count} expedited shipments`}
          trend={{
            value: Math.abs(expeditedShipping.trend),
            label: "vs last month",
            isPositive: true,
          }}
          icon={<FiTruck />}
          colorScheme="green"
        />

        <MetricCard
          title="Customer Penalties"
          value={`$${customerPenalties.totalCost.toLocaleString()}`}
          subtitle={`${customerPenalties.count} late/incomplete deliveries`}
          trend={{
            value: Math.abs(customerPenalties.trend),
            label: "vs last month",
            isPositive: true,
          }}
          icon={<FiDollarSign />}
          colorScheme="green"
        />

        <MetricCard
          title="Total Problem Costs"
          value={`$${totalMonthlyCosts.toLocaleString()}`}
          subtitle="All issue-related costs this month"
          trend={{
            value: costReduction,
            label: "reduction",
            isPositive: true,
          }}
          icon={<FiTrendingDown />}
          colorScheme="green"
        />
      </SimpleGrid>

      <Grid templateColumns={{ base: "1fr", lg: "1fr 1fr" }} gap={6}>
        <StatChart
          title="Monthly Problem Costs Trend"
          subtitle="Cost reduction over time by category"
          type="area"
          data={monthlyProblemCosts}
          dataKey="rejections"
          xAxisKey="name"
          height={300}
          colorScheme="#e53e3e"
        >
          {/* TODO: Add multiple area series for expedited and penalties */}
        </StatChart>

        <StatChart
          title="Cost Breakdown by Issue Type"
          subtitle="Where problems are costing the most"
          type="bar"
          data={costByIssueType}
          dataKey="cost"
          xAxisKey="name"
          height={300}
          colorScheme="#ed8936"
          hideLegend={true}
        />
      </Grid>
    </Grid>
  )
}

export default ProblemCosts
