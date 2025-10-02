import { Box, Flex, Input, Text, Badge } from "@chakra-ui/react"
import { useState } from "react"
import { FiVideo, FiSearch } from "react-icons/fi"
import type { Camera, CameraZone } from "./types"

interface CameraListProps {
  zones: CameraZone[]
  onCameraSelect?: (camera: Camera) => void
  selectedCameraIds?: Set<string>
  onCameraDragStart?: (camera: Camera) => void
}

function CameraList({ zones, onCameraSelect, selectedCameraIds = new Set(), onCameraDragStart }: CameraListProps) {
  const [searchQuery, setSearchQuery] = useState("")

  const getStatusColor = (status: Camera["status"]) => {
    switch (status) {
      case "online":
      case "recording":
        return "green"
      case "offline":
        return "red"
      case "buffering":
        return "yellow"
      case "error":
        return "red"
      default:
        return "gray"
    }
  }

  const filteredZones = zones.map((zone) => ({
    ...zone,
    cameras: zone.cameras.filter(
      (camera) =>
        camera.name.toLowerCase().includes(searchQuery.toLowerCase()) ||
        camera.location.toLowerCase().includes(searchQuery.toLowerCase())
    ),
  })).filter((zone) => zone.cameras.length > 0)

  return (
    <Box>
      {/* Search Bar */}
      <Box mb={4}>
        <Flex align="center" gap={2} px={3} py={2} bg="bg.muted" borderRadius="md">
          <FiSearch />
          <Input
            placeholder="Search cameras..."
            value={searchQuery}
            onChange={(e) => setSearchQuery(e.target.value)}
            variant="outline"
            size="sm"
          />
        </Flex>
      </Box>

      {/* Camera Groups by Zone */}
      <Box>
        {filteredZones.map((zone) => (
          <Box key={zone.id} mb={4}>
            <Text fontSize="xs" fontWeight="bold" px={2} py={1} color="fg.subtle">
              {zone.name}
            </Text>
            <Box>
              {zone.cameras.map((camera) => {
                const isSelected = selectedCameraIds.has(camera.id)
                return (
                  <Flex
                    key={camera.id}
                    align="center"
                    gap={3}
                    px={3}
                    py={2}
                    cursor="grab"
                    draggable
                    bg={isSelected ? "bg.muted" : "transparent"}
                    _hover={{ bg: "bg.muted" }}
                    borderRadius="md"
                    onClick={() => onCameraSelect?.(camera)}
                    onDragStart={(e) => {
                      onCameraDragStart?.(camera)
                      if (e.currentTarget) {
                        e.currentTarget.style.cursor = "grabbing"
                      }
                    }}
                    onDragEnd={(e) => {
                      if (e.currentTarget) {
                        e.currentTarget.style.cursor = "grab"
                      }
                    }}
                  >
                    <FiVideo />
                    <Box flex="1" minW={0}>
                      <Text fontSize="sm" fontWeight="medium" truncate>
                        {camera.name}
                      </Text>
                      <Text fontSize="xs" color="fg.subtle" truncate>
                        {camera.location}
                      </Text>
                    </Box>
                    <Badge colorPalette={getStatusColor(camera.status)} size="sm">
                      {camera.status}
                    </Badge>
                  </Flex>
                )
              })}
            </Box>
          </Box>
        ))}

        {filteredZones.length === 0 && (
          <Text fontSize="sm" color="fg.subtle" textAlign="center" py={4}>
            No cameras found
          </Text>
        )}
      </Box>
    </Box>
  )
}

export default CameraList
