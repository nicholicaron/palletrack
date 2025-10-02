import { Grid, SimpleGrid } from "@chakra-ui/react"
import { FiDollarSign, FiTrendingDown, FiTrendingUp, FiUsers } from "react-icons/fi"

import MetricCard from "../MetricCard"
import StatChart from "../StatChart"

const CostBenefit = () => {
  // TODO: Fetch from API - GET /api/financial/labor-savings
  const laborSavings = {
    hoursPerDay: 12.5,
    hoursPerWeek: 87.5,
    hoursPerMonth: 375,
    costPerHour: 25,
    dailySavings: 312.5,
    weeklySavings: 2187.5,
    monthlySavings: 9375,
  }

  // TODO: Fetch from API - GET /api/financial/rejection-reduction
  const rejectionReduction = {
    beforeAutomation: 45,
    afterAutomation: 8,
    reductionPercentage: 82,
    avgCostPerRejection: 150,
    monthlySavings: 5550,
  }

  // TODO: Fetch from API - GET /api/financial/detention-fees
  const detentionFees = {
    previousMonth: 4500,
    currentMonth: 800,
    reduction: 82,
    savings: 3700,
  }

  // TODO: Fetch from API - GET /api/financial/customer-satisfaction
  const customerSatisfaction = {
    score: 94,
    improvement: 15,
    delayedShipments: 3,
    onTimePercentage: 98.5,
  }

  // TODO: Fetch from API - GET /api/financial/monthly-comparison
  const monthlyComparison = [
    { name: "Jan", manual: 28000, automated: 12000 },
    { name: "Feb", manual: 29000, automated: 11500 },
    { name: "Mar", manual: 30000, automated: 11000 },
    { name: "Apr", manual: 31000, automated: 10500 },
  ]

  // TODO: Fetch from API - GET /api/financial/savings-breakdown
  const savingsBreakdown = [
    { name: "Labor", value: 9375 },
    { name: "Rejections", value: 5550 },
    { name: "Detention Fees", value: 3700 },
    { name: "Expedited Shipping", value: 2200 },
  ]

  const totalMonthlySavings =
    laborSavings.monthlySavings +
    rejectionReduction.monthlySavings +
    detentionFees.savings +
    2200

  return (
    <Grid gap={6}>
      <SimpleGrid columns={{ base: 1, md: 2, lg: 4 }} gap={4}>
        <MetricCard
          title="Labor Hours Saved"
          value={`${laborSavings.hoursPerMonth}h/mo`}
          subtitle={`$${laborSavings.monthlySavings.toLocaleString()} saved`}
          trend={{
            value: 100,
            label: "vs manual",
            isPositive: true,
          }}
          icon={<FiUsers />}
          colorScheme="green"
        />

        <MetricCard
          title="Rejection Reduction"
          value={`${rejectionReduction.reductionPercentage}%`}
          subtitle={`$${rejectionReduction.monthlySavings.toLocaleString()}/mo saved`}
          trend={{
            value: rejectionReduction.reductionPercentage,
            label: "reduction",
            isPositive: true,
          }}
          icon={<FiTrendingDown />}
          colorScheme="green"
        />

        <MetricCard
          title="Detention Fees Avoided"
          value={`$${detentionFees.savings.toLocaleString()}`}
          subtitle={`${detentionFees.reduction}% reduction this month`}
          trend={{
            value: detentionFees.reduction,
            label: "reduction",
            isPositive: true,
          }}
          icon={<FiDollarSign />}
          colorScheme="green"
        />

        <MetricCard
          title="Customer Satisfaction"
          value={`${customerSatisfaction.score}%`}
          subtitle={`${customerSatisfaction.onTimePercentage}% on-time delivery`}
          trend={{
            value: customerSatisfaction.improvement,
            label: "improvement",
            isPositive: true,
          }}
          icon={<FiTrendingUp />}
          colorScheme="green"
        />
      </SimpleGrid>

      <Grid templateColumns={{ base: "1fr", lg: "1fr 1fr" }} gap={6}>
        <StatChart
          title="Monthly Cost Comparison"
          subtitle="Manual vs Automated processing costs"
          type="line"
          data={monthlyComparison}
          dataKey="manual"
          dataKey2="automated"
          xAxisKey="name"
          height={300}
          colorScheme="#e53e3e"
        >
          {/* TODO: Add second line series for automated costs */}
        </StatChart>

        <StatChart
          title="Monthly Savings Breakdown"
          subtitle={`Total savings: $${totalMonthlySavings.toLocaleString()}`}
          type="pie"
          data={savingsBreakdown}
          dataKey="value"
          xAxisKey="name"
          height={300}
          colorScheme="#48bb78"
        />
      </Grid>
    </Grid>
  )
}

export default CostBenefit
