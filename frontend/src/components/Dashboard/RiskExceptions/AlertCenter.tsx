import { Box, Collapsible, Grid, Stack } from "@chakra-ui/react"
import { useState } from "react"
import { FiAlertCircle, FiAlertTriangle, FiChevronDown, FiChevronUp, FiPackage } from "react-icons/fi"

import AlertBadge from "../AlertBadge"

const AlertCenter = () => {
  const [expandedCategory, setExpandedCategory] = useState<string | null>(null)

  // TODO: Fetch from API - GET /api/alerts/active
  const activeAlerts = [
    {
      id: 1,
      severity: "critical" as const,
      message: "3 shipments departing in 15 minutes without complete documentation",
      count: 3,
    },
    {
      id: 2,
      severity: "warning" as const,
      message: "5 pallets sitting idle for over 2 hours without scanning",
      count: 5,
    },
    {
      id: 3,
      severity: "critical" as const,
      message: "Hazmat shipment #HZ-2847 requires additional verification",
      count: 1,
    },
    {
      id: 4,
      severity: "warning" as const,
      message: "High-value shipment (>$50k) flagged for additional review",
      count: 2,
    },
    {
      id: 5,
      severity: "info" as const,
      message: "12 shipments completed and ready for pickup",
      count: 12,
    },
    {
      id: 6,
      severity: "critical" as const,
      message: "Pallet #P-8392 missing BOL documentation",
      count: 1,
    },
    {
      id: 7,
      severity: "warning" as const,
      message: "OCR confidence below 80% on 4 documents",
      count: 4,
    },
    {
      id: 8,
      severity: "warning" as const,
      message: "Dock 3 lighting causing readability issues",
      count: 1,
    },
    {
      id: 9,
      severity: "info" as const,
      message: "Morning shift completed 156 pallets",
      count: 1,
    },
    {
      id: 10,
      severity: "info" as const,
      message: "Peak hour processing reached 89 pallets/hour",
      count: 1,
    },
  ]

  // TODO: Fetch from API - GET /api/alerts/summary
  // Alert counts are now calculated dynamically from activeAlerts by filtering severity

  const getIcon = (severity: string) => {
    switch (severity) {
      case "critical":
        return <FiAlertCircle />
      case "warning":
        return <FiAlertTriangle />
      default:
        return <FiPackage />
    }
  }

  const toggleCategory = (category: string) => {
    setExpandedCategory(expandedCategory === category ? null : category)
  }

  const renderAlertCategory = (
    category: "critical" | "warning" | "info",
    bgColor: string,
    borderColor: string,
    textColor: string,
    label: string,
    icon: React.ReactNode
  ) => {
    const categoryAlerts = activeAlerts.filter((a) => a.severity === category)
    const isExpanded = expandedCategory === category

    return (
      <Box>
        <Box
          p={3}
          bg={bgColor}
          borderWidth="1px"
          borderColor={borderColor}
          borderRadius="md"
          cursor="pointer"
          _hover={{ shadow: "md" }}
          transition="all 0.2s"
          onClick={() => toggleCategory(category)}
        >
          <Grid templateColumns="1fr auto" alignItems="center" gap={2}>
            <Box display="flex" alignItems="center" gap={3}>
              <Box fontSize="xl" color={textColor}>
                {icon}
              </Box>
              <Box>
                <Box fontSize="2xl" fontWeight="bold" color={textColor}>
                  {categoryAlerts.length}
                </Box>
                <Box fontSize="sm" color={textColor}>
                  {label}
                </Box>
              </Box>
            </Box>
            <Box fontSize="xl" color={textColor}>
              {isExpanded ? <FiChevronUp /> : <FiChevronDown />}
            </Box>
          </Grid>
        </Box>

        <Collapsible.Root open={isExpanded}>
          <Collapsible.Content>
            <Box
              mt={2}
              maxH="200px"
              overflowY="auto"
              bg="white"
              borderWidth="1px"
              borderColor="gray.200"
              borderRadius="md"
              p={2}
              css={{
                "&::-webkit-scrollbar": {
                  width: "8px",
                },
                "&::-webkit-scrollbar-track": {
                  background: "#f1f1f1",
                  borderRadius: "4px",
                },
                "&::-webkit-scrollbar-thumb": {
                  background: "#888",
                  borderRadius: "4px",
                },
                "&::-webkit-scrollbar-thumb:hover": {
                  background: "#555",
                },
              }}
            >
              <Stack gap={2}>
                {categoryAlerts.map((alert) => (
                  <AlertBadge
                    key={alert.id}
                    severity={alert.severity}
                    message={alert.message}
                    count={alert.count}
                    icon={getIcon(alert.severity)}
                    onClick={() => {
                      // TODO: Implement alert drill-down navigation
                      console.log(`Navigate to alert details: ${alert.id}`)
                    }}
                  />
                ))}
              </Stack>
            </Box>
          </Collapsible.Content>
        </Collapsible.Root>
      </Box>
    )
  }

  return (
    <Grid gap={4}>
      <Grid templateColumns={{ base: "1fr", md: "repeat(3, 1fr)" }} gap={4}>
        {renderAlertCategory(
          "critical",
          "red.50",
          "red.200",
          "red.700",
          "Critical Alerts",
          <FiAlertCircle />
        )}
        {renderAlertCategory(
          "warning",
          "yellow.50",
          "yellow.200",
          "yellow.700",
          "Warnings",
          <FiAlertTriangle />
        )}
        {renderAlertCategory(
          "info",
          "blue.50",
          "blue.200",
          "blue.700",
          "Info Notifications",
          <FiPackage />
        )}
      </Grid>
    </Grid>
  )
}

export default AlertCenter
