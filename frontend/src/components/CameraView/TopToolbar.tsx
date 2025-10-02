import { Flex } from "@chakra-ui/react"
import type { GridLayoutType, SequenceConfig, ViewPreset } from "./types"
import LayoutSelector from "./LayoutSelector"
import ZoneSelector from "./ZoneSelector"
import PresetSelector from "./PresetSelector"
import SequenceModeToggle from "./SequenceModeToggle"

interface TopToolbarProps {
  layoutType: GridLayoutType
  onLayoutChange: (layout: GridLayoutType) => void
  selectedZone?: string
  onZoneChange: (zone: string | undefined) => void
  selectedPreset?: string
  onPresetChange: (presetId: string | undefined) => void
  sequenceConfig: SequenceConfig
  onSequenceConfigChange: (config: SequenceConfig) => void
  zones: string[]
  presets: ViewPreset[]
  onSavePreset: () => void
  onResetView: () => void
}

function TopToolbar({
  layoutType,
  onLayoutChange,
  selectedZone,
  onZoneChange,
  selectedPreset,
  onPresetChange,
  sequenceConfig,
  onSequenceConfigChange,
  zones,
  presets,
  onSavePreset,
  onResetView,
}: TopToolbarProps) {
  return (
    <Flex
      justify="space-between"
      align="center"
      px={4}
      py={2}
      bg="bg.subtle"
      borderBottom="1px solid"
      borderColor="border"
    >
      <Flex gap={2} align="center">
        <ZoneSelector
          zones={zones}
          selectedZone={selectedZone}
          onZoneChange={onZoneChange}
        />
        <PresetSelector
          presets={presets}
          selectedPresetId={selectedPreset}
          onPresetChange={onPresetChange}
          onSavePreset={onSavePreset}
          onResetView={onResetView}
        />
        <SequenceModeToggle
          config={sequenceConfig}
          onConfigChange={onSequenceConfigChange}
        />
      </Flex>
      <Flex gap={2} align="center">
        <LayoutSelector
          selectedLayout={layoutType}
          onLayoutChange={onLayoutChange}
        />
      </Flex>
    </Flex>
  )
}

export default TopToolbar
