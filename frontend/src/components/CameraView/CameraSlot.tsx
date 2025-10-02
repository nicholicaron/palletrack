import { Box, Flex, Text, Badge } from "@chakra-ui/react"
import { FiVideo, FiRefreshCw } from "react-icons/fi"
import { useState } from "react"
import type { Camera } from "./types"

interface CameraSlotProps {
  camera: Camera | null
  isSelected?: boolean
  onSelect?: () => void
  onDoubleClick?: () => void
  onContextMenu?: (e: React.MouseEvent) => void
  onDragStart?: () => void
  onDragOver?: (e: React.DragEvent) => void
  onDrop?: () => void
}

function CameraSlot({
  camera,
  isSelected = false,
  onSelect,
  onDoubleClick,
  onContextMenu,
  onDragStart,
  onDragOver,
  onDrop,
}: CameraSlotProps) {
  const [isDragOver, setIsDragOver] = useState(false)
  const getStatusColor = (status: Camera["status"]) => {
    switch (status) {
      case "online":
        return "blue"
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

  return (
    <Box
      position="relative"
      h="full"
      bg="bg.muted"
      borderRadius="md"
      border="2px solid"
      borderColor={isDragOver ? "blue.400" : isSelected ? "blue.500" : "border"}
      overflow="hidden"
      cursor="grab"
      draggable
      onClick={onSelect}
      onDoubleClick={onDoubleClick}
      onContextMenu={onContextMenu}
      onDragStart={(e) => {
        onDragStart?.()
        if (e.currentTarget) {
          e.currentTarget.style.cursor = "grabbing"
          // Create a smaller drag image
          const canvas = document.createElement('canvas')
          canvas.width = 150
          canvas.height = 100
          const ctx = canvas.getContext('2d')
          if (ctx) {
            ctx.fillStyle = '#2D3748'
            ctx.fillRect(0, 0, 150, 100)
            ctx.fillStyle = '#CBD5E0'
            ctx.font = '12px sans-serif'
            ctx.textAlign = 'center'
            ctx.fillText(camera?.name || 'Camera', 75, 50)
          }
          e.dataTransfer.setDragImage(canvas, 75, 50)
        }
      }}
      onDragEnd={(e) => {
        if (e.currentTarget) {
          e.currentTarget.style.cursor = "grab"
        }
      }}
      onDragOver={(e) => {
        setIsDragOver(true)
        onDragOver?.(e)
      }}
      onDragLeave={() => {
        setIsDragOver(false)
      }}
      onDrop={(e) => {
        e.preventDefault()
        setIsDragOver(false)
        onDrop?.()
      }}
      transition="all 0.2s"
      _hover={{
        borderColor: isSelected ? "blue.600" : "border.emphasized",
        transform: "scale(1.02)",
      }}
    >
      {/* Drag Over Overlay */}
      {isDragOver && (
        <Box
          position="absolute"
          top={0}
          left={0}
          right={0}
          bottom={0}
          bg="blue.500"
          opacity={0.2}
          zIndex={10}
          display="flex"
          alignItems="center"
          justifyContent="center"
        >
          <Flex
            direction="column"
            align="center"
            gap={2}
            bg="blackAlpha.900"
            px={6}
            py={4}
            borderRadius="md"
            zIndex={11}
          >
            <FiRefreshCw size={32} color="white" />
            <Text color="white" fontWeight="bold" fontSize="sm">
              Release to Swap
            </Text>
          </Flex>
        </Box>
      )}
      {camera ? (
        <>
          {/* Camera Header */}
          <Flex
            position="absolute"
            top={0}
            left={0}
            right={0}
            justify="space-between"
            align="center"
            px={2}
            py={1}
            bg="blackAlpha.700"
            zIndex={1}
          >
            <Text fontSize="xs" fontWeight="medium" color="white" truncate>
              {camera.name}
            </Text>
            <Badge colorPalette={getStatusColor(camera.status)} size="xs">
              {camera.status}
            </Badge>
          </Flex>

          {/* Video Placeholder */}
          <Flex
            h="full"
            align="center"
            justify="center"
            direction="column"
            gap={2}
            bg="gray.900"
          >
            <FiVideo size={32} color="gray" />
            <Text fontSize="xs" color="gray.400">
              Stream will appear here
            </Text>
          </Flex>

          {/* Camera Info Footer */}
          <Flex
            position="absolute"
            bottom={0}
            left={0}
            right={0}
            px={2}
            py={1}
            bg="blackAlpha.700"
            zIndex={1}
          >
            <Text fontSize="xs" color="gray.300" truncate>
              {camera.location}
            </Text>
          </Flex>
        </>
      ) : (
        /* Empty Slot */
        <Flex
          h="full"
          align="center"
          justify="center"
          direction="column"
          gap={2}
        >
          <FiVideo size={24} color="gray" />
          <Text fontSize="xs" color="fg.subtle">
            No camera assigned
          </Text>
        </Flex>
      )}
    </Box>
  )
}

export default CameraSlot
