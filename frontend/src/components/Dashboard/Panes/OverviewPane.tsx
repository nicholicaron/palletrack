import { Grid, SimpleGrid } from "@chakra-ui/react"

import AlertCenter from "../RiskExceptions/AlertCenter"
import ThroughputPanel from "../PrimaryMetrics/ThroughputPanel"
import ShipmentStatus from "../PrimaryMetrics/ShipmentStatus"

const OverviewPane = () => {
  return (
    <Grid gap={6} h="full">
      {/* Top Row - Alerts */}
      <AlertCenter />

      {/* Bottom Row - Split between Throughput and Status */}
      <SimpleGrid columns={{ base: 1, xl: 2 }} gap={6}>
        <ThroughputPanel />
        <ShipmentStatus />
      </SimpleGrid>
    </Grid>
  )
}

export default OverviewPane
