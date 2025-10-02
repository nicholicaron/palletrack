import { Box, Card, Heading, Text } from "@chakra-ui/react"
import type { ReactNode } from "react"
import {
  Area,
  AreaChart,
  Bar,
  BarChart,
  CartesianGrid,
  Cell,
  Legend,
  Line,
  LineChart,
  Pie,
  PieChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts"

export type ChartType = "line" | "bar" | "area" | "pie"

export interface StatChartProps {
  title: string
  subtitle?: string
  type: ChartType
  data: any[]
  dataKey?: string
  dataKey2?: string
  xAxisKey?: string
  height?: number
  colorScheme?: string
  hideLegend?: boolean
  useSingleColor?: boolean
  children?: ReactNode
}

const CHART_COLORS = [
  "#3182ce", // blue
  "#48bb78", // green
  "#ed8936", // orange
  "#9f7aea", // purple
  "#38b2ac", // teal
  "#e53e3e", // red
  "#d69e2e", // yellow
  "#805ad5", // violet
  "#dd6b20", // dark orange
  "#319795", // cyan
]

const StatChart = ({
  title,
  subtitle,
  type,
  data,
  dataKey = "value",
  dataKey2,
  xAxisKey = "name",
  height = 300,
  colorScheme = "#3182ce",
  hideLegend = false,
  useSingleColor = false,
  children,
}: StatChartProps) => {
  const renderChart = () => {
    const commonProps = {
      data,
      margin: { top: 5, right: 20, left: 0, bottom: 40 },
    }

    switch (type) {
      case "line":
        return (
          <LineChart {...commonProps}>
            <CartesianGrid strokeDasharray="3 3" stroke="#e2e8f0" />
            <XAxis
              dataKey={xAxisKey}
              stroke="#718096"
              fontSize={11}
              angle={-90}
              textAnchor="end"
              height={80}
            />
            <YAxis stroke="#718096" fontSize={12} />
            <Tooltip
              contentStyle={{
                backgroundColor: "#ffffff",
                border: "1px solid #cbd5e0",
                borderRadius: "8px",
                padding: "12px",
                boxShadow: "0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06)",
              }}
              itemStyle={{
                color: "#2d3748",
                fontSize: "14px",
                fontWeight: "500",
              }}
              labelStyle={{
                color: "#1a202c",
                fontWeight: "600",
                marginBottom: "4px",
              }}
            />
            {!hideLegend && <Legend />}
            <Line
              type="monotone"
              dataKey={dataKey}
              stroke={colorScheme}
              strokeWidth={2}
              dot={{ fill: colorScheme }}
            />
            {dataKey2 && (
              <Line
                type="monotone"
                dataKey={dataKey2}
                stroke={CHART_COLORS[1]}
                strokeWidth={2}
                dot={{ fill: CHART_COLORS[1] }}
              />
            )}
          </LineChart>
        )

      case "bar":
        return (
          <BarChart {...commonProps}>
            <CartesianGrid strokeDasharray="3 3" stroke="#e2e8f0" />
            <XAxis
              dataKey={xAxisKey}
              stroke="#718096"
              fontSize={11}
              angle={-90}
              textAnchor="end"
              height={80}
            />
            <YAxis stroke="#718096" fontSize={12} />
            <Tooltip
              contentStyle={{
                backgroundColor: "#ffffff",
                border: "1px solid #cbd5e0",
                borderRadius: "8px",
                padding: "12px",
                boxShadow: "0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06)",
              }}
              itemStyle={{
                color: "#2d3748",
                fontSize: "14px",
                fontWeight: "500",
              }}
              labelStyle={{
                color: "#1a202c",
                fontWeight: "600",
                marginBottom: "4px",
              }}
            />
            {!hideLegend && <Legend />}
            <Bar dataKey={dataKey} radius={[4, 4, 0, 0]}>
              {useSingleColor
                ? data.map((_entry, index) => (
                    <Cell key={`cell-${index}`} fill={colorScheme} />
                  ))
                : data.map((_entry, index) => (
                    <Cell key={`cell-${index}`} fill={CHART_COLORS[index % CHART_COLORS.length]} />
                  ))
              }
            </Bar>
          </BarChart>
        )

      case "area":
        return (
          <AreaChart {...commonProps}>
            <CartesianGrid strokeDasharray="3 3" stroke="#e2e8f0" />
            <XAxis
              dataKey={xAxisKey}
              stroke="#718096"
              fontSize={11}
              angle={-90}
              textAnchor="end"
              height={80}
            />
            <YAxis stroke="#718096" fontSize={12} />
            <Tooltip
              contentStyle={{
                backgroundColor: "#ffffff",
                border: "1px solid #cbd5e0",
                borderRadius: "8px",
                padding: "12px",
                boxShadow: "0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06)",
              }}
              itemStyle={{
                color: "#2d3748",
                fontSize: "14px",
                fontWeight: "500",
              }}
              labelStyle={{
                color: "#1a202c",
                fontWeight: "600",
                marginBottom: "4px",
              }}
            />
            {!hideLegend && <Legend />}
            <Area
              type="monotone"
              dataKey={dataKey}
              stroke={colorScheme}
              fill={colorScheme}
              fillOpacity={0.3}
            />
          </AreaChart>
        )

      case "pie":
        return (
          <PieChart>
            <Pie
              data={data}
              dataKey={dataKey}
              nameKey={xAxisKey}
              cx="50%"
              cy="50%"
              outerRadius={100}
              label
            >
              {data.map((_entry, index) => (
                <Cell key={`cell-${index}`} fill={CHART_COLORS[index % CHART_COLORS.length]} />
              ))}
            </Pie>
            <Tooltip
              contentStyle={{
                backgroundColor: "#ffffff",
                border: "1px solid #cbd5e0",
                borderRadius: "8px",
                padding: "12px",
                boxShadow: "0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06)",
              }}
              itemStyle={{
                color: "#2d3748",
                fontSize: "14px",
                fontWeight: "500",
              }}
              labelStyle={{
                color: "#1a202c",
                fontWeight: "600",
                marginBottom: "4px",
              }}
            />
            <Legend />
          </PieChart>
        )

      default:
        return null
    }
  }

  return (
    <Card.Root p={3} borderWidth="1px" borderColor="gray.200">
      <Card.Body p={0}>
        <Box mb={3}>
          <Heading size="md" mb={1}>
            {title}
          </Heading>
          {subtitle && (
            <Text fontSize="sm" color="gray.600">
              {subtitle}
            </Text>
          )}
        </Box>

        <ResponsiveContainer width="100%" height={height}>
          {renderChart() || <div />}
        </ResponsiveContainer>

        {children && <Box mt={4}>{children}</Box>}
      </Card.Body>
    </Card.Root>
  )
}

export default StatChart
