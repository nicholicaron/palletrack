import { Grid, Progress, SimpleGrid } from "@chakra-ui/react"
import { FiAlertTriangle, FiCheckCircle, FiFileText, FiShield } from "react-icons/fi"

import MetricCard from "../MetricCard"
import StatChart from "../StatChart"

const ComplianceTracker = () => {
  // TODO: Fetch from API - GET /api/compliance/hazmat
  const hazmatCompliance = {
    total: 15,
    compliant: 14,
    percentage: 93.3,
  }

  // TODO: Fetch from API - GET /api/compliance/temperature-controlled
  const tempControlledCompliance = {
    total: 8,
    compliant: 8,
    percentage: 100,
  }

  // TODO: Fetch from API - GET /api/compliance/restricted-goods
  const restrictedGoodsCompliance = {
    total: 6,
    compliant: 5,
    percentage: 83.3,
  }

  // TODO: Fetch from API - GET /api/compliance/upcoming-deadlines
  const upcomingDeadlines = [
    {
      id: 1,
      item: "Hazmat Carrier Certification",
      daysRemaining: 15,
    },
    {
      id: 2,
      item: "DOT Annual Inspection",
      daysRemaining: 42,
    },
    {
      id: 3,
      item: "Alcohol Shipping License",
      daysRemaining: 68,
    },
  ]

  // TODO: Fetch from API - GET /api/compliance/trends
  const complianceTrends = [
    { name: "Week 1", hazmat: 90, temp: 98, restricted: 85 },
    { name: "Week 2", hazmat: 92, temp: 100, restricted: 88 },
    { name: "Week 3", hazmat: 91, temp: 97, restricted: 82 },
    { name: "Week 4", hazmat: 93, temp: 100, restricted: 83 },
  ]

  return (
    <Grid gap={6}>
      <SimpleGrid columns={{ base: 1, md: 2, lg: 4 }} gap={4}>
        <MetricCard
          title="Hazmat Compliance"
          value={`${hazmatCompliance.percentage}%`}
          subtitle={`${hazmatCompliance.compliant} of ${hazmatCompliance.total} shipments`}
          icon={<FiAlertTriangle />}
          colorScheme={
            hazmatCompliance.percentage >= 95
              ? "green"
              : hazmatCompliance.percentage >= 70
              ? "yellow"
              : "red"
          }
        >
          <Progress.Root
            value={hazmatCompliance.percentage}
            size="sm"
            colorPalette={
              hazmatCompliance.percentage >= 95
                ? "green"
                : hazmatCompliance.percentage >= 70
                ? "yellow"
                : "red"
            }
          >
            <Progress.Track>
              <Progress.Range />
            </Progress.Track>
          </Progress.Root>
        </MetricCard>

        <MetricCard
          title="Temp-Controlled"
          value={`${tempControlledCompliance.percentage}%`}
          subtitle={`${tempControlledCompliance.compliant} of ${tempControlledCompliance.total} shipments`}
          icon={<FiCheckCircle />}
          colorScheme={
            tempControlledCompliance.percentage >= 95
              ? "green"
              : tempControlledCompliance.percentage >= 70
              ? "yellow"
              : "red"
          }
        >
          <Progress.Root
            value={tempControlledCompliance.percentage}
            size="sm"
            colorPalette={
              tempControlledCompliance.percentage >= 95
                ? "green"
                : tempControlledCompliance.percentage >= 70
                ? "yellow"
                : "red"
            }
          >
            <Progress.Track>
              <Progress.Range />
            </Progress.Track>
          </Progress.Root>
        </MetricCard>

        <MetricCard
          title="Restricted Goods"
          value={`${restrictedGoodsCompliance.percentage}%`}
          subtitle={`${restrictedGoodsCompliance.compliant} of ${restrictedGoodsCompliance.total} shipments`}
          icon={<FiShield />}
          colorScheme={
            restrictedGoodsCompliance.percentage >= 95
              ? "green"
              : restrictedGoodsCompliance.percentage >= 70
              ? "yellow"
              : "red"
          }
        >
          <Progress.Root
            value={restrictedGoodsCompliance.percentage}
            size="sm"
            colorPalette={
              restrictedGoodsCompliance.percentage >= 95
                ? "green"
                : restrictedGoodsCompliance.percentage >= 70
                ? "yellow"
                : "red"
            }
          >
            <Progress.Track>
              <Progress.Range />
            </Progress.Track>
          </Progress.Root>
        </MetricCard>

        <MetricCard
          title="Next Certification Due"
          value={`${upcomingDeadlines[0].daysRemaining} days`}
          subtitle={upcomingDeadlines[0].item}
          icon={<FiFileText />}
          colorScheme={upcomingDeadlines[0].daysRemaining < 30 ? "yellow" : "blue"}
        />
      </SimpleGrid>

      <StatChart
        title="Regulatory Compliance Trends"
        subtitle="Compliance percentage by category over time"
        type="line"
        data={complianceTrends}
        dataKey="hazmat"
        xAxisKey="name"
        height={300}
        colorScheme="#e53e3e"
      >
        {/* TODO: Add multiple line series for temp and restricted */}
      </StatChart>
    </Grid>
  )
}

export default ComplianceTracker
