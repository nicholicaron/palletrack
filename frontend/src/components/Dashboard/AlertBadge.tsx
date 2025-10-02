import { Badge, Box, Flex, Text } from "@chakra-ui/react"
import type { ReactNode } from "react"

export type AlertSeverity = "info" | "warning" | "critical" | "success"

export interface AlertBadgeProps {
  severity: AlertSeverity
  message: string
  count?: number
  icon?: ReactNode
  onClick?: () => void
}

const AlertBadge = ({
  severity,
  message,
  count,
  icon,
  onClick,
}: AlertBadgeProps) => {
  const getSeverityConfig = () => {
    switch (severity) {
      case "critical":
        return {
          colorScheme: "red",
          bg: "red.50",
          border: "red.300",
          text: "red.800",
        }
      case "warning":
        return {
          colorScheme: "yellow",
          bg: "yellow.50",
          border: "yellow.300",
          text: "yellow.800",
        }
      case "success":
        return {
          colorScheme: "green",
          bg: "green.50",
          border: "green.300",
          text: "green.800",
        }
      default:
        return {
          colorScheme: "blue",
          bg: "blue.50",
          border: "blue.300",
          text: "blue.800",
        }
    }
  }

  const config = getSeverityConfig()

  return (
    <Box
      p={3}
      bg={config.bg}
      borderWidth="1px"
      borderColor={config.border}
      borderRadius="md"
      cursor={onClick ? "pointer" : "default"}
      onClick={onClick}
      _hover={onClick ? { shadow: "sm", borderColor: config.text } : {}}
      transition="all 0.2s"
    >
      <Flex align="center" gap={3}>
        {icon && (
          <Box color={config.text} fontSize="lg">
            {icon}
          </Box>
        )}

        <Box flex="1">
          <Text fontSize="sm" fontWeight="medium" color={config.text}>
            {message}
          </Text>
        </Box>

        {count !== undefined && count > 0 && (
          <Badge colorScheme={config.colorScheme} fontSize="xs" px={2} py={1}>
            {count}
          </Badge>
        )}
      </Flex>
    </Box>
  )
}

export default AlertBadge
