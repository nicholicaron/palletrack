import { Grid } from "@chakra-ui/react"

import TeamPerformance from "../WorkforceManagement/TeamPerformance"
import ResourceAllocation from "../WorkforceManagement/ResourceAllocation"

const WorkforcePane = () => {
  return (
    <Grid gap={6} h="full" templateColumns={{ base: "1fr", xl: "1fr 1fr" }}>
      <TeamPerformance />
      <ResourceAllocation />
    </Grid>
  )
}

export default WorkforcePane
