import { Grid, Box } from "@chakra-ui/react"
import type { GridLayoutType, Camera } from "./types"
import CameraSlot from "./CameraSlot"

interface GridLayoutProps {
  layoutType: GridLayoutType
  cameras: (Camera | null)[]
  selectedSlots: Set<number>
  onSlotSelect: (index: number, ctrlKey: boolean) => void
  onSlotDoubleClick: (index: number) => void
  onDragStart: (index: number) => void
  onDragOver: (e: React.DragEvent) => void
  onDrop: (index: number) => void
}

function GridLayout({
  layoutType,
  cameras,
  selectedSlots,
  onSlotSelect,
  onSlotDoubleClick,
  onDragStart,
  onDragOver,
  onDrop,
}: GridLayoutProps) {
  const getGridConfig = () => {
    switch (layoutType) {
      case "1x1":
        return { columns: 1, rows: 1, slots: 1 }
      case "2x2":
        return { columns: 2, rows: 2, slots: 4 }
      case "3x3":
        return { columns: 3, rows: 3, slots: 9 }
      case "4x4":
        return { columns: 4, rows: 4, slots: 16 }
      case "2x3":
        return { columns: 3, rows: 2, slots: 6 }
      case "1+5":
      case "1+7":
        return { columns: 1, rows: 1, slots: layoutType === "1+5" ? 6 : 8, special: true }
      default:
        return { columns: 4, rows: 4, slots: 16 }
    }
  }

  const config = getGridConfig()

  // Special layouts (1+5, 1+7)
  if (config.special) {
    const mainCamera = cameras[0]
    const thumbnailCameras = cameras.slice(1, config.slots)

    if (layoutType === "1+5") {
      return (
        <Grid templateColumns="1fr 200px" gap={2} h="full">
          {/* Main large view */}
          <Box>
            <CameraSlot
              camera={mainCamera}
              isSelected={selectedSlots.has(0)}
              onSelect={() => onSlotSelect(0, false)}
              onDoubleClick={() => onSlotDoubleClick(0)}
              onDragStart={() => onDragStart(0)}
              onDragOver={onDragOver}
              onDrop={() => onDrop(0)}
            />
          </Box>
          {/* 5 thumbnails on right */}
          <Grid templateRows="repeat(5, 1fr)" gap={2}>
            {thumbnailCameras.map((camera, i) => (
              <CameraSlot
                key={i + 1}
                camera={camera}
                isSelected={selectedSlots.has(i + 1)}
                onSelect={() => onSlotSelect(i + 1, false)}
                onDoubleClick={() => onSlotDoubleClick(i + 1)}
                onDragStart={() => onDragStart(i + 1)}
                onDragOver={onDragOver}
                onDrop={() => onDrop(i + 1)}
              />
            ))}
          </Grid>
        </Grid>
      )
    }

    if (layoutType === "1+7") {
      return (
        <Grid templateRows="1fr 150px" gap={2} h="full">
          <Grid templateColumns="1fr 200px" gap={2}>
            {/* Main large view */}
            <Box>
              <CameraSlot
                camera={mainCamera}
                isSelected={selectedSlots.has(0)}
                onSelect={() => onSlotSelect(0, false)}
                onDoubleClick={() => onSlotDoubleClick(0)}
                onDragStart={() => onDragStart(0)}
                onDragOver={onDragOver}
                onDrop={() => onDrop(0)}
              />
            </Box>
            {/* 3 thumbnails on right */}
            <Grid templateRows="repeat(3, 1fr)" gap={2}>
              {thumbnailCameras.slice(0, 3).map((camera, i) => (
                <CameraSlot
                  key={i + 1}
                  camera={camera}
                  isSelected={selectedSlots.has(i + 1)}
                  onSelect={() => onSlotSelect(i + 1, false)}
                  onDoubleClick={() => onSlotDoubleClick(i + 1)}
                  onDragStart={() => onDragStart(i + 1)}
                  onDragOver={onDragOver}
                  onDrop={() => onDrop(i + 1)}
                />
              ))}
            </Grid>
          </Grid>
          {/* 4 thumbnails on bottom */}
          <Grid templateColumns="repeat(4, 1fr)" gap={2}>
            {thumbnailCameras.slice(3, 7).map((camera, i) => (
              <CameraSlot
                key={i + 4}
                camera={camera}
                isSelected={selectedSlots.has(i + 4)}
                onSelect={() => onSlotSelect(i + 4, false)}
                onDoubleClick={() => onSlotDoubleClick(i + 4)}
                onDragStart={() => onDragStart(i + 4)}
                onDragOver={onDragOver}
                onDrop={() => onDrop(i + 4)}
              />
            ))}
          </Grid>
        </Grid>
      )
    }
  }

  // Standard grid layouts
  return (
    <Grid
      templateColumns={`repeat(${config.columns}, 1fr)`}
      templateRows={`repeat(${config.rows}, 1fr)`}
      gap={2}
      h="full"
    >
      {Array.from({ length: config.slots }).map((_, index) => {
        const camera = cameras[index] || null
        return (
          <CameraSlot
            key={index}
            camera={camera}
            isSelected={selectedSlots.has(index)}
            onSelect={() => onSlotSelect(index, false)}
            onDoubleClick={() => onSlotDoubleClick(index)}
            onDragStart={() => onDragStart(index)}
            onDragOver={onDragOver}
            onDrop={() => onDrop(index)}
          />
        )
      })}
    </Grid>
  )
}

export default GridLayout
