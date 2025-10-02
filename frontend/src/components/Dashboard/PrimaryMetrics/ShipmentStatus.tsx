import { Grid, SimpleGrid } from "@chakra-ui/react"
import { FiAlertTriangle, FiCheckCircle, FiClock, FiPackage } from "react-icons/fi"

import MetricCard from "../MetricCard"
import StatChart from "../StatChart"

const ShipmentStatus = () => {
  // TODO: Fetch from API - GET /api/metrics/shipments/status
  const shipmentStatus = {
    readyToShip: 156,
    pendingReview: 12,
    flaggedIssues: 3,
    inLoadingBay: 23,
  }

  // TODO: Fetch from API - GET /api/metrics/shipments/breakdown
  const statusBreakdown = [
    { name: "Ready to Ship", value: shipmentStatus.readyToShip },
    { name: "Pending Review", value: shipmentStatus.pendingReview },
    { name: "Flagged", value: shipmentStatus.flaggedIssues },
    { name: "In Bay", value: shipmentStatus.inLoadingBay },
  ]

  const totalPallets =
    shipmentStatus.readyToShip +
    shipmentStatus.pendingReview +
    shipmentStatus.flaggedIssues +
    shipmentStatus.inLoadingBay

  return (
    <Grid gap={6}>
      <SimpleGrid columns={{ base: 1, md: 2, lg: 4 }} gap={4}>
        <MetricCard
          title="Ready to Ship"
          value={shipmentStatus.readyToShip}
          subtitle="All docs verified and complete"
          icon={<FiCheckCircle />}
          colorScheme="green"
        />

        <MetricCard
          title="Pending Review"
          value={shipmentStatus.pendingReview}
          subtitle="Low confidence or missing info"
          icon={<FiClock />}
          colorScheme="yellow"
        />

        <MetricCard
          title="Flagged for Issues"
          value={shipmentStatus.flaggedIssues}
          subtitle="Requires immediate attention"
          icon={<FiAlertTriangle />}
          colorScheme="red"
        />

        <MetricCard
          title="In Loading Bay"
          value={shipmentStatus.inLoadingBay}
          subtitle="Currently being processed"
          icon={<FiPackage />}
          colorScheme="blue"
        />
      </SimpleGrid>

      <StatChart
        title="Shipment Status Breakdown"
        subtitle={`Total pallets: ${totalPallets}`}
        type="pie"
        data={statusBreakdown}
        dataKey="value"
        xAxisKey="name"
        height={300}
        colorScheme="#805ad5"
      />
    </Grid>
  )
}

export default ShipmentStatus
