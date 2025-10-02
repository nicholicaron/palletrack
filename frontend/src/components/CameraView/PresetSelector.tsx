import { Button } from "../ui/button"
import {
  MenuContent,
  MenuItem,
  MenuRoot,
  MenuSeparator,
  MenuTrigger,
} from "../ui/menu"
import { FiSave, FiStar, FiRotateCcw } from "react-icons/fi"
import type { ViewPreset } from "./types"

interface PresetSelectorProps {
  presets: ViewPreset[]
  selectedPresetId?: string
  onPresetChange: (presetId: string | undefined) => void
  onSavePreset?: () => void
  onResetView?: () => void
}

function PresetSelector({
  presets,
  selectedPresetId,
  onPresetChange,
  onSavePreset,
  onResetView,
}: PresetSelectorProps) {
  const systemPresets = presets.filter((p) => p.isSystemPreset)
  const userPresets = presets.filter((p) => !p.isSystemPreset)
  const selectedPreset = presets.find((p) => p.id === selectedPresetId)

  return (
    <MenuRoot>
      <MenuTrigger asChild>
        <Button variant="outline" size="sm">
          <FiStar />
          {selectedPreset?.name || "Select Preset"}
        </Button>
      </MenuTrigger>
      <MenuContent>
        {onSavePreset && (
          <>
            <MenuItem value="save" onClick={onSavePreset}>
              <FiSave /> Save Current View
            </MenuItem>
            <MenuSeparator />
          </>
        )}

        {onResetView && (
          <>
            <MenuItem value="reset" onClick={onResetView}>
              <FiRotateCcw /> Reset to Original
            </MenuItem>
            <MenuSeparator />
          </>
        )}

        <MenuItem
          value="none"
          onClick={() => onPresetChange(undefined)}
        >
          No Preset {!selectedPresetId && "✓"}
        </MenuItem>

        {systemPresets.length > 0 && (
          <>
            <MenuSeparator />
            {systemPresets.map((preset) => (
              <MenuItem
                key={preset.id}
                value={preset.id}
                onClick={() => onPresetChange(preset.id)}
              >
                {preset.name} ({preset.layoutType})
                {selectedPresetId === preset.id && " ✓"}
              </MenuItem>
            ))}
          </>
        )}

        {userPresets.length > 0 && (
          <>
            <MenuSeparator />
            {userPresets.map((preset) => (
              <MenuItem
                key={preset.id}
                value={preset.id}
                onClick={() => {
                  onPresetChange(preset.id)
                }}
              >
                {preset.name} ({preset.layoutType})
                {selectedPresetId === preset.id && " ✓"}
              </MenuItem>
            ))}
          </>
        )}
      </MenuContent>
    </MenuRoot>
  )
}

export default PresetSelector
