import { Grid } from "@chakra-ui/react"

import ComplianceTracker from "../RiskExceptions/ComplianceTracker"
import DocumentationCompliance from "../OperationalInsights/DocumentationCompliance"
import CarrierPerformance from "../CarrierRouting/CarrierPerformance"

const CompliancePane = () => {
  return (
    <Grid gap={6} h="full">
      <Grid templateColumns={{ base: "1fr", xl: "1fr 1fr" }} gap={6}>
        <ComplianceTracker />
        <CarrierPerformance />
      </Grid>
      <DocumentationCompliance />
    </Grid>
  )
}

export default CompliancePane
