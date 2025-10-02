import { IconButton, Flex, Text } from "@chakra-ui/react"
import { FiPlay, FiPause } from "react-icons/fi"
import {
  MenuContent,
  MenuItem,
  MenuRoot,
  MenuSeparator,
  MenuTrigger,
} from "../ui/menu"
import type { SequenceConfig } from "./types"

interface SequenceModeToggleProps {
  config: SequenceConfig
  onConfigChange: (config: SequenceConfig) => void
}

const dwellTimeOptions = [5, 10, 15, 30]

function SequenceModeToggle({ config, onConfigChange }: SequenceModeToggleProps) {
  const toggleEnabled = () => {
    onConfigChange({ ...config, enabled: !config.enabled })
  }

  const setDwellTime = (seconds: number) => {
    onConfigChange({ ...config, dwellTime: seconds })
  }

  return (
    <Flex align="center" gap={2}>
      <IconButton
        variant={config.enabled ? "solid" : "outline"}
        colorScheme={config.enabled ? "blue" : "gray"}
        size="sm"
        onClick={toggleEnabled}
        aria-label={config.enabled ? "Stop sequence" : "Start sequence"}
      >
        {config.enabled ? <FiPause /> : <FiPlay />}
      </IconButton>

      {config.enabled && (
        <MenuRoot>
          <MenuTrigger asChild>
            <Text
              fontSize="sm"
              cursor="pointer"
              px={2}
              py={1}
              borderRadius="md"
              _hover={{ bg: "bg.muted" }}
            >
              {config.dwellTime}s
            </Text>
          </MenuTrigger>
          <MenuContent>
            <Text fontSize="xs" px={2} py={1} fontWeight="bold" color="fg.subtle">
              Dwell Time
            </Text>
            <MenuSeparator />
            {dwellTimeOptions.map((seconds) => (
              <MenuItem
                key={seconds}
                value={String(seconds)}
                onClick={() => setDwellTime(seconds)}
              >
                {seconds} seconds {config.dwellTime === seconds && "✓"}
              </MenuItem>
            ))}
          </MenuContent>
        </MenuRoot>
      )}
    </Flex>
  )
}

export default SequenceModeToggle
