import { Grid } from "@chakra-ui/react"

import AccuracyScore from "../PrimaryMetrics/AccuracyScore"
import SpeedEfficiency from "../OperationalInsights/SpeedEfficiency"
import QualityControl from "../OperationalInsights/QualityControl"

const OperationsPane = () => {
  return (
    <Grid gap={6} h="full" templateRows="auto 1fr">
      {/* Top - Accuracy Metrics */}
      <AccuracyScore />

      {/* Bottom - Split between Speed and Quality */}
      <Grid templateColumns={{ base: "1fr", xl: "1fr 1fr" }} gap={6}>
        <SpeedEfficiency />
        <QualityControl />
      </Grid>
    </Grid>
  )
}

export default OperationsPane
