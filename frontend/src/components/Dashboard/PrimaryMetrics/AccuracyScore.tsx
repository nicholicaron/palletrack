import { Box, Grid, Progress, SimpleGrid, Text } from "@chakra-ui/react"
import { FiAlertCircle, FiCheckCircle, FiFileText, FiTrendingUp } from "react-icons/fi"

import MetricCard from "../MetricCard"
import StatChart from "../StatChart"

const AccuracyScore = () => {
  // TODO: Fetch from API - GET /api/metrics/accuracy/document-completeness
  const documentCompleteness = {
    percentage: 94.5,
    palletsComplete: 323,
    palletsMissing: 19,
    trend: 2.3,
  }

  // TODO: Fetch from API - GET /api/metrics/accuracy/ocr-confidence
  const ocrConfidence = {
    averageScore: 97.2,
    trend: 1.5,
  }

  // TODO: Fetch from API - GET /api/metrics/accuracy/missing-docs
  const missingDocs = {
    count: 3,
    types: ["BOL", "Packing List"],
  }

  // TODO: Fetch from API - GET /api/metrics/accuracy/trends
  const accuracyTrendData = [
    { name: "Mon", score: 92 },
    { name: "Tue", score: 94 },
    { name: "Wed", score: 93 },
    { name: "Thu", score: 95 },
    { name: "Fri", score: 96 },
    { name: "Sat", score: 94 },
    { name: "Sun", score: 95 },
  ]

  return (
    <Grid gap={6}>
      <SimpleGrid columns={{ base: 1, md: 2, lg: 4 }} gap={4}>
        <MetricCard
          title="Documentation Complete"
          value={`${documentCompleteness.percentage}%`}
          subtitle={`${documentCompleteness.palletsComplete} of ${documentCompleteness.palletsComplete + documentCompleteness.palletsMissing} pallets`}
          trend={{
            value: documentCompleteness.trend,
            label: "vs last week",
            isPositive: true,
          }}
          icon={<FiFileText />}
          colorScheme={
            documentCompleteness.percentage >= 95
              ? "green"
              : documentCompleteness.percentage >= 70
              ? "yellow"
              : "red"
          }
        >
          <Progress.Root
            value={documentCompleteness.percentage}
            size="sm"
            colorPalette={
              documentCompleteness.percentage >= 95
                ? "green"
                : documentCompleteness.percentage >= 70
                ? "yellow"
                : "red"
            }
          >
            <Progress.Track>
              <Progress.Range />
            </Progress.Track>
          </Progress.Root>
        </MetricCard>

        <MetricCard
          title="OCR Confidence Score"
          value={`${ocrConfidence.averageScore}%`}
          subtitle="Average document readability"
          trend={{
            value: ocrConfidence.trend,
            label: "vs last week",
            isPositive: true,
          }}
          icon={<FiCheckCircle />}
          colorScheme={
            ocrConfidence.averageScore >= 95
              ? "green"
              : ocrConfidence.averageScore >= 70
              ? "yellow"
              : "red"
          }
        >
          <Progress.Root
            value={ocrConfidence.averageScore}
            size="sm"
            colorPalette={
              ocrConfidence.averageScore >= 95
                ? "green"
                : ocrConfidence.averageScore >= 70
                ? "yellow"
                : "red"
            }
          >
            <Progress.Track>
              <Progress.Range />
            </Progress.Track>
          </Progress.Root>
        </MetricCard>

        <MetricCard
          title="Missing Documents"
          value={missingDocs.count}
          subtitle="Pallets shipped without complete docs"
          icon={<FiAlertCircle />}
          colorScheme={missingDocs.count > 5 ? "red" : "yellow"}
        >
          <Box>
            <Text fontSize="xs" color="gray.600">
              Types: {missingDocs.types.join(", ")}
            </Text>
          </Box>
        </MetricCard>

        <MetricCard
          title="Quality Trend"
          value="Improving"
          subtitle="Week-over-week accuracy"
          icon={<FiTrendingUp />}
          colorScheme="green"
        />
      </SimpleGrid>

      <StatChart
        title="Document Quality Trends"
        subtitle="Average accuracy score over time"
        type="line"
        data={accuracyTrendData}
        dataKey="score"
        xAxisKey="name"
        height={250}
        colorScheme="#48bb78"
      />
    </Grid>
  )
}

export default AccuracyScore
