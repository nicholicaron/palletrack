import { Grid } from "@chakra-ui/react"

import PerformanceOverTime from "../HistoricalTrends/PerformanceOverTime"
import ProblemPatterns from "../HistoricalTrends/ProblemPatterns"
import DestinationAnalytics from "../CarrierRouting/DestinationAnalytics"

const PerformancePane = () => {
  return (
    <Grid gap={6} h="full">
      <Grid templateColumns={{ base: "1fr", xl: "1fr 1fr" }} gap={6}>
        <PerformanceOverTime />
        <ProblemPatterns />
      </Grid>
      <DestinationAnalytics />
    </Grid>
  )
}

export default PerformancePane
