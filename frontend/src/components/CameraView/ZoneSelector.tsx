import { Button } from "../ui/button"
import {
  MenuContent,
  MenuItem,
  MenuRoot,
  MenuTrigger,
} from "../ui/menu"
import { FiMapPin } from "react-icons/fi"

interface ZoneSelectorProps {
  zones: string[]
  selectedZone?: string
  onZoneChange: (zone: string | undefined) => void
}

function ZoneSelector({ zones, selectedZone, onZoneChange }: ZoneSelectorProps) {
  return (
    <MenuRoot>
      <MenuTrigger asChild>
        <Button variant="outline" size="sm">
          <FiMapPin />
          {selectedZone || "All Zones"}
        </Button>
      </MenuTrigger>
      <MenuContent>
        <MenuItem
          value="all"
          onClick={() => onZoneChange(undefined)}
        >
          All Zones {!selectedZone && "✓"}
        </MenuItem>
        {zones.map((zone) => (
          <MenuItem
            key={zone}
            value={zone}
            onClick={() => onZoneChange(zone)}
          >
            {zone} {selectedZone === zone && "✓"}
          </MenuItem>
        ))}
      </MenuContent>
    </MenuRoot>
  )
}

export default ZoneSelector
