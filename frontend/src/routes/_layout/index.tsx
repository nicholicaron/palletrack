import { Box, Container, Tabs } from "@chakra-ui/react"
import { createFileRoute } from "@tanstack/react-router"
import { useState } from "react"

import OverviewPane from "@/components/Dashboard/Panes/OverviewPane"
import OperationsPane from "@/components/Dashboard/Panes/OperationsPane"
import CompliancePane from "@/components/Dashboard/Panes/CompliancePane"
import PerformancePane from "@/components/Dashboard/Panes/PerformancePane"
import FinancialPane from "@/components/Dashboard/Panes/FinancialPane"
import WorkforcePane from "@/components/Dashboard/Panes/WorkforcePane"
import InsightsPane from "@/components/Dashboard/Panes/InsightsPane"

export const Route = createFileRoute("/_layout/")({
  component: Dashboard,
})

function Dashboard() {
  const [activeTab, setActiveTab] = useState("overview")

  const panes = [
    { id: "overview", label: "Overview", component: OverviewPane },
    { id: "operations", label: "Operations", component: OperationsPane },
    { id: "compliance", label: "Compliance & Risk", component: CompliancePane },
    { id: "performance", label: "Performance", component: PerformancePane },
    { id: "financial", label: "Financial", component: FinancialPane },
    { id: "workforce", label: "Workforce", component: WorkforcePane },
    { id: "insights", label: "Insights", component: InsightsPane },
  ]

  const ActivePaneComponent = panes.find((p) => p.id === activeTab)?.component || OverviewPane

  return (
    <Container maxW="full" h="calc(100vh - 80px)" p={4}>
      <Tabs.Root
        value={activeTab}
        onValueChange={(e) => setActiveTab(e.value)}
        variant="enclosed"
        size="lg"
        h="full"
      >
        <Tabs.List mb={4}>
          {panes.map((pane) => (
            <Tabs.Trigger key={pane.id} value={pane.id}>
              {pane.label}
            </Tabs.Trigger>
          ))}
        </Tabs.List>

        <Box h="calc(100% - 60px)" overflowY="auto">
          <ActivePaneComponent />
        </Box>
      </Tabs.Root>
    </Container>
  )
}
