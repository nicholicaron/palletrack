import { Flex, Text, Badge } from "@chakra-ui/react"
import type { SystemStatus } from "./types"

interface StatusBarProps {
  status: SystemStatus
}

function StatusBar({ status }: StatusBarProps) {
  return (
    <Flex
      justify="space-between"
      align="center"
      px={4}
      py={2}
      bg="bg.subtle"
      borderBottom="1px solid"
      borderColor="border"
      fontSize="sm"
    >
      <Flex gap={6} align="center">
        <Flex gap={2} align="center">
          <Text fontWeight="medium">Cameras:</Text>
          <Badge colorPalette="green">{status.onlineCameras} Online</Badge>
          <Badge colorPalette="red">{status.offlineCameras} Offline</Badge>
          <Badge colorPalette="blue">{status.recordingCameras} Recording</Badge>
        </Flex>

        <Text>
          <Text as="span" fontWeight="medium">Bandwidth:</Text>{" "}
          {status.totalBandwidth}
        </Text>

        <Text>
          <Text as="span" fontWeight="medium">Storage:</Text>{" "}
          {status.storageRemaining}
        </Text>
      </Flex>

      <Flex gap={4} align="center">
        {status.activeAlerts > 0 && (
          <Badge colorPalette="orange" cursor="pointer">
            {status.activeAlerts} Alert{status.activeAlerts > 1 ? "s" : ""}
          </Badge>
        )}

        <Text color="fg.subtle">
          {new Date().toLocaleTimeString()}
        </Text>
      </Flex>
    </Flex>
  )
}

export default StatusBar
