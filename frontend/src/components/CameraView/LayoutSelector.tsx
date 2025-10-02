import { IconButton, Flex, Box } from "@chakra-ui/react"
import { HiOutlineViewGrid } from "react-icons/hi"
import {
  MenuContent,
  MenuItem,
  MenuRoot,
  MenuTrigger,
} from "../ui/menu"
import type { GridLayoutType } from "./types"

interface LayoutSelectorProps {
  selectedLayout: GridLayoutType
  onLayoutChange: (layout: GridLayoutType) => void
}

const layoutOptions: Array<{ type: GridLayoutType; label: string; icon: string }> = [
  { type: "1x1", label: "Single View", icon: "□" },
  { type: "2x2", label: "Quad View (2x2)", icon: "⊞" },
  { type: "3x3", label: "9 Camera Grid (3x3)", icon: "⊟" },
  { type: "4x4", label: "16 Camera Grid (4x4)", icon: "⊠" },
  { type: "1+5", label: "1 Large + 5 Small", icon: "⊡" },
  { type: "1+7", label: "1 Large + 7 Small", icon: "⊞" },
  { type: "2x3", label: "6 Camera Grid (2x3)", icon: "⊟" },
]

function LayoutSelector({ selectedLayout, onLayoutChange }: LayoutSelectorProps) {
  return (
    <MenuRoot>
      <MenuTrigger asChild>
        <IconButton
          variant="outline"
          size="sm"
          aria-label="Select grid layout"
        >
          <HiOutlineViewGrid />
        </IconButton>
      </MenuTrigger>
      <MenuContent>
        {layoutOptions.map((option) => (
          <MenuItem
            key={option.type}
            value={option.type}
            onClick={() => onLayoutChange(option.type)}
          >
            <Flex align="center" gap={3} w="full">
              <Box fontSize="xl" fontFamily="monospace">
                {option.icon}
              </Box>
              <Box flex="1">
                {option.label}
                {option.type === selectedLayout && " ✓"}
              </Box>
            </Flex>
          </MenuItem>
        ))}
      </MenuContent>
    </MenuRoot>
  )
}

export default LayoutSelector
