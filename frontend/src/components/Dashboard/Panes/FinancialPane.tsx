import { Grid } from "@chakra-ui/react"

import CostBenefit from "../FinancialImpact/CostBenefit"
import ProblemCosts from "../FinancialImpact/ProblemCosts"

const FinancialPane = () => {
  return (
    <Grid gap={6} h="full" templateColumns={{ base: "1fr", xl: "1fr 1fr" }}>
      <CostBenefit />
      <ProblemCosts />
    </Grid>
  )
}

export default FinancialPane
