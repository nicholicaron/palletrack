import { Box, Card, Flex, Stat, Text } from "@chakra-ui/react"
import type { ReactNode } from "react"

export interface MetricCardProps {
  title: string
  value: string | number
  subtitle?: string
  trend?: {
    value: number
    label: string
    isPositive?: boolean
  }
  icon?: ReactNode
  colorScheme?: "green" | "yellow" | "red" | "blue" | "gray"
  children?: ReactNode
}

const MetricCard = ({
  title,
  value,
  subtitle,
  trend,
  icon,
  colorScheme = "blue",
  children,
}: MetricCardProps) => {
  const getColorScheme = () => {
    switch (colorScheme) {
      case "green":
        return { bg: "green.50", border: "green.200", accent: "green.600" }
      case "yellow":
        return { bg: "yellow.50", border: "yellow.200", accent: "yellow.600" }
      case "red":
        return { bg: "red.50", border: "red.200", accent: "red.600" }
      case "blue":
        return { bg: "blue.50", border: "blue.200", accent: "blue.600" }
      default:
        return { bg: "gray.50", border: "gray.200", accent: "gray.600" }
    }
  }

  const colors = getColorScheme()

  return (
    <Card.Root
      p={3}
      borderWidth="1px"
      borderColor={colors.border}
      bg={colors.bg}
      _hover={{ shadow: "md" }}
      transition="all 0.2s"
      h="full"
    >
      <Card.Body p={0}>
        <Flex justify="space-between" align="flex-start" mb={2} gap={2}>
          <Box flex="1" minW={0}>
            <Text
              fontSize="xs"
              fontWeight="medium"
              color="gray.600"
              mb={1}
              lineHeight="1.3"
            >
              {title}
            </Text>
            <Stat.Root>
              <Stat.ValueText
                fontSize="2xl"
                fontWeight="bold"
                color={colors.accent}
                lineHeight="1.2"
                css={{
                  wordBreak: "break-word",
                }}
              >
                {value}
              </Stat.ValueText>
            </Stat.Root>
          </Box>
          {icon && (
            <Box color={colors.accent} fontSize="xl" flexShrink={0}>
              {icon}
            </Box>
          )}
        </Flex>

        {subtitle && (
          <Text
            fontSize="xs"
            color="gray.500"
            mb={2}
            lineHeight="1.3"
            lineClamp={2}
          >
            {subtitle}
          </Text>
        )}

        {trend && (
          <Flex align="center" gap={2} flexWrap="wrap">
            <Text
              fontSize="xs"
              fontWeight="semibold"
              color={trend.isPositive ? "green.600" : "red.600"}
              whiteSpace="nowrap"
            >
              {trend.isPositive ? "↑" : "↓"} {Math.abs(trend.value)}%
            </Text>
            <Text fontSize="xs" color="gray.500" lineClamp={1}>
              {trend.label}
            </Text>
          </Flex>
        )}

        {children && <Box mt={2}>{children}</Box>}
      </Card.Body>
    </Card.Root>
  )
}

export default MetricCard
