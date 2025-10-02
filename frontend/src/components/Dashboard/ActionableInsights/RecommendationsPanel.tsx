import { Box, Card, Flex, Grid, Heading, Stack, Text } from "@chakra-ui/react"
import { FiAlertCircle, FiInfo, FiTrendingUp } from "react-icons/fi"

interface Recommendation {
  id: number
  type: "critical" | "improvement" | "optimization" | "info"
  title: string
  description: string
  impact: "high" | "medium" | "low"
  category: string
}

const RecommendationsPanel = () => {
  // TODO: Fetch from API - GET /api/insights/recommendations
  const recommendations: Recommendation[] = [
    {
      id: 1,
      type: "critical",
      title: "Improve Lighting in Dock 3",
      description:
        "OCR accuracy is 40% lower in this location. Installing additional LED lighting could improve readability scores and reduce manual verification.",
      impact: "high",
      category: "Infrastructure",
    },
    {
      id: 2,
      type: "improvement",
      title: "Train Afternoon Shift on BOL Requirements",
      description:
        "Afternoon shift shows 3x higher missing document rate. Targeted training session recommended for BOL compliance.",
      impact: "high",
      category: "Training",
    },
    {
      id: 3,
      type: "critical",
      title: "Contact Purchasing About Vendor XYZ",
      description:
        "Vendor XYZ consistently ships without packing lists (34 incidents this month). Recommend vendor compliance discussion.",
      impact: "medium",
      category: "Vendor Management",
    },
    {
      id: 4,
      type: "optimization",
      title: "Add Scanning Station for Peak Hours",
      description:
        "Peak activity 2-4 PM shows bottleneck. Additional scanning station could increase throughput by 25%.",
      impact: "high",
      category: "Capacity",
    },
    {
      id: 5,
      type: "improvement",
      title: "Standardize Label Format for Regional LTL",
      description:
        "Regional LTL carrier has 8.5% rejection rate due to label format issues. Work with carrier to standardize format.",
      impact: "medium",
      category: "Carrier Relations",
    },
    {
      id: 6,
      type: "info",
      title: "SKU-8472 Requires Special Handling",
      description:
        "This SKU has 23 documentation issues. Consider creating a special handling checklist for this product.",
      impact: "low",
      category: "Process",
    },
  ]

  // TODO: Fetch from API - GET /api/insights/document-templates
  const documentTemplateStats = {
    totalFormats: 47,
    problematicFormats: 8,
    standardizationOpportunity: 17,
  }

  const getTypeIcon = (type: string) => {
    switch (type) {
      case "critical":
        return <FiAlertCircle />
      case "improvement":
        return <FiTrendingUp />
      case "optimization":
        return <FiTrendingUp />
      default:
        return <FiInfo />
    }
  }

  const getTypeColor = (type: string) => {
    switch (type) {
      case "critical":
        return { bg: "red.50", border: "red.300", text: "red.800" }
      case "improvement":
        return { bg: "yellow.50", border: "yellow.300", text: "yellow.800" }
      case "optimization":
        return { bg: "blue.50", border: "blue.300", text: "blue.800" }
      default:
        return { bg: "gray.50", border: "gray.300", text: "gray.800" }
    }
  }

  const getImpactBadge = (impact: string) => {
    const colors = {
      high: "red",
      medium: "yellow",
      low: "gray",
    }
    return (
      <Text
        fontSize="xs"
        fontWeight="bold"
        px={2}
        py={1}
        bg={`${colors[impact as keyof typeof colors]}.100`}
        color={`${colors[impact as keyof typeof colors]}.800`}
        borderRadius="md"
      >
        {impact.toUpperCase()} IMPACT
      </Text>
    )
  }

  return (
    <Grid gap={6}>
      <Box>
        <Heading size="lg" mb={2}>
          Recommendations
        </Heading>
        <Text color="gray.600" fontSize="sm">
          AI-powered insights to improve operations
        </Text>
      </Box>

      <Stack gap={4}>
        {recommendations.map((rec) => {
          const colors = getTypeColor(rec.type)
          return (
            <Card.Root
              key={rec.id}
              p={4}
              borderWidth="1px"
              borderColor={colors.border}
              bg={colors.bg}
              _hover={{ shadow: "md" }}
              transition="all 0.2s"
            >
              <Card.Body>
                <Flex gap={4} align="flex-start">
                  <Box color={colors.text} fontSize="2xl" mt={1}>
                    {getTypeIcon(rec.type)}
                  </Box>
                  <Box flex="1">
                    <Flex justify="space-between" align="flex-start" mb={2}>
                      <Box>
                        <Heading size="sm" mb={1} color={colors.text}>
                          {rec.title}
                        </Heading>
                        <Text fontSize="xs" color="gray.600" mb={2}>
                          {rec.category}
                        </Text>
                      </Box>
                      {getImpactBadge(rec.impact)}
                    </Flex>
                    <Text fontSize="sm" color="gray.700">
                      {rec.description}
                    </Text>
                  </Box>
                </Flex>
              </Card.Body>
            </Card.Root>
          )
        })}
      </Stack>

      {/* Document Template Library Stats */}
      <Card.Root p={5} borderWidth="1px" borderColor="gray.200">
        <Card.Body>
          <Heading size="md" mb={4}>
            Document Template Library
          </Heading>
          <Grid templateColumns={{ base: "1fr", md: "repeat(3, 1fr)" }} gap={4}>
            <Box>
              <Text fontSize="2xl" fontWeight="bold" color="blue.600">
                {documentTemplateStats.totalFormats}
              </Text>
              <Text fontSize="sm" color="gray.600">
                Document formats scanned
              </Text>
            </Box>
            <Box>
              <Text fontSize="2xl" fontWeight="bold" color="red.600">
                {documentTemplateStats.problematicFormats}
              </Text>
              <Text fontSize="sm" color="gray.600">
                Formats causing OCR issues
              </Text>
            </Box>
            <Box>
              <Text fontSize="2xl" fontWeight="bold" color="yellow.600">
                {documentTemplateStats.standardizationOpportunity}
              </Text>
              <Text fontSize="sm" color="gray.600">
                Opportunities for standardization
              </Text>
            </Box>
          </Grid>
          <Box mt={4} p={3} bg="blue.50" borderRadius="md">
            <Text fontSize="sm" color="blue.800">
              <strong>Suggestion:</strong> Work with top 5 vendors to
              standardize document formats. This could reduce OCR errors by an
              estimated 35%.
            </Text>
          </Box>
        </Card.Body>
      </Card.Root>
    </Grid>
  )
}

export default RecommendationsPanel
