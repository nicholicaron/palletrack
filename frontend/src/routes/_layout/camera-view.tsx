import { Box, Flex, Text, Badge } from "@chakra-ui/react"
import { createFileRoute } from "@tanstack/react-router"
import { useState, useMemo, useEffect } from "react"
import { FiVideo } from "react-icons/fi"

import StatusBar from "@/components/CameraView/StatusBar"
import TopToolbar from "@/components/CameraView/TopToolbar"
import LeftSidebar from "@/components/CameraView/LeftSidebar"
import CameraList from "@/components/CameraView/CameraList"
import GridLayout from "@/components/CameraView/GridLayout"
import PaginationControls from "@/components/CameraView/PaginationControls"
import { generateMockCameraZones, generateMockCameras64 } from "@/components/CameraView/mockData"
import type { GridLayoutType, SequenceConfig, ViewPreset, SystemStatus, Camera } from "@/components/CameraView/types"

export const Route = createFileRoute("/_layout/camera-view")({
  component: CameraView,
})

function CameraView() {
  const [sidebarOpen, setSidebarOpen] = useState(true)
  const [layoutType, setLayoutType] = useState<GridLayoutType>("1+5")
  const [selectedZone, setSelectedZone] = useState<string | undefined>()
  const [selectedPresetId, setSelectedPresetId] = useState<string | undefined>()
  const [selectedSlots, setSelectedSlots] = useState<Set<number>>(new Set())
  const [currentPage, setCurrentPage] = useState(0)
  const [fullScreenCamera, setFullScreenCamera] = useState<number | null>(null)
  const [draggedSlot, setDraggedSlot] = useState<number | null>(null)
  const [draggedCamera, setDraggedCamera] = useState<Camera | null>(null)
  const [cameraAssignments, setCameraAssignments] = useState<Map<number, string>>(new Map())
  const [savedPresets, setSavedPresets] = useState<ViewPreset[]>([])
  const [sequenceConfig, setSequenceConfig] = useState<SequenceConfig>({
    enabled: false,
    dwellTime: 10,
    mode: "all-cameras",
    pauseOnMotion: false,
  })

  // Load saved presets from localStorage on mount
  useEffect(() => {
    const stored = localStorage.getItem('cameraViewPresets')
    if (stored) {
      try {
        const parsed = JSON.parse(stored)
        const presets = parsed.map((p: any) => ({
          ...p,
          cameraAssignments: new Map(Object.entries(p.cameraAssignments || {})),
          createdAt: new Date(p.createdAt),
        }))
        setSavedPresets(presets)
      } catch (e) {
        console.error('Failed to load presets:', e)
      }
    }
  }, [])

  // Save presets to localStorage whenever they change
  useEffect(() => {
    if (savedPresets.length > 0) {
      const serializable = savedPresets.map(p => ({
        ...p,
        cameraAssignments: Object.fromEntries(p.cameraAssignments),
        createdAt: p.createdAt.toISOString(),
      }))
      localStorage.setItem('cameraViewPresets', JSON.stringify(serializable))
    }
  }, [savedPresets])

  // Mock data
  const cameraZones = useMemo(() => generateMockCameraZones(), [])
  const allCameras = useMemo(() => generateMockCameras64(), [])

  // Filter cameras by zone
  const filteredCameras = useMemo(() => {
    if (!selectedZone) return allCameras
    return allCameras.filter((cam) => cam.zone === selectedZone)
  }, [allCameras, selectedZone])

  // Get cameras per page based on layout
  const getCamerasPerPage = () => {
    switch (layoutType) {
      case "1x1": return 1
      case "2x2": return 4
      case "3x3": return 9
      case "4x4": return 16
      case "2x3": return 6
      case "1+5": return 6
      case "1+7": return 8
      default: return 16
    }
  }

  const camerasPerPage = getCamerasPerPage()
  const totalPages = Math.ceil(filteredCameras.length / camerasPerPage)
  const startIndex = currentPage * camerasPerPage
  const endIndex = Math.min(startIndex + camerasPerPage, filteredCameras.length)
  const currentPageCameras = filteredCameras.slice(startIndex, endIndex)

  // Apply custom camera assignments, then pad with nulls
  const assignedCameras = Array(camerasPerPage).fill(null).map((_, index) => {
    const globalIndex = startIndex + index
    const assignedCameraId = cameraAssignments.get(globalIndex)

    if (assignedCameraId) {
      return allCameras.find(c => c.id === assignedCameraId) || null
    }

    return currentPageCameras[index] || null
  })

  // System status
  const systemStatus: SystemStatus = useMemo(() => {
    const onlineCameras = allCameras.filter(c => c.status === "online" || c.status === "recording").length
    const offlineCameras = allCameras.filter(c => c.status === "offline").length
    const recordingCameras = allCameras.filter(c => c.status === "recording").length

    return {
      totalCameras: allCameras.length,
      onlineCameras,
      offlineCameras,
      recordingCameras,
      totalBandwidth: "125 Mbps",
      storageRemaining: "2.4 TB",
      activeAlerts: 0,
    }
  }, [allCameras])

  const zones = Array.from(new Set(allCameras.map(c => c.zone)))

  const handleSavePreset = () => {
    const name = prompt('Enter a name for this preset:')
    if (!name) return

    const newPreset: ViewPreset = {
      id: `preset-${Date.now()}`,
      name,
      layoutType,
      cameraAssignments: new Map(cameraAssignments),
      createdAt: new Date(),
      isSystemPreset: false,
    }

    setSavedPresets([...savedPresets, newPreset])
  }

  const handleLoadPreset = (presetId: string | undefined) => {
    setSelectedPresetId(presetId)

    if (!presetId) return

    const preset = savedPresets.find(p => p.id === presetId)
    if (preset) {
      setLayoutType(preset.layoutType)
      setCameraAssignments(new Map(preset.cameraAssignments))
      setCurrentPage(0)
    }
  }

  const handleResetView = () => {
    setCameraAssignments(new Map())
    setSelectedPresetId(undefined)
  }

  const handleSlotSelect = (index: number, ctrlKey: boolean) => {
    const newSelected = new Set(selectedSlots)

    if (ctrlKey) {
      // Multi-select: toggle
      if (newSelected.has(index)) {
        newSelected.delete(index)
      } else {
        newSelected.add(index)
      }
    } else {
      // Single select
      newSelected.clear()
      newSelected.add(index)
    }

    setSelectedSlots(newSelected)
  }

  const handleSlotDoubleClick = (index: number) => {
    const camera = assignedCameras[index]
    if (camera) {
      setFullScreenCamera(index)
    }
  }

  const handleExitFullScreen = () => {
    setFullScreenCamera(null)
  }

  const handleFullScreenKeyPress = (e: KeyboardEvent) => {
    if (e.key === "Escape" && fullScreenCamera !== null) {
      handleExitFullScreen()
    }
  }

  // Add keyboard listener for ESC key
  useEffect(() => {
    if (fullScreenCamera !== null) {
      window.addEventListener("keydown", handleFullScreenKeyPress)
      return () => window.removeEventListener("keydown", handleFullScreenKeyPress)
    }
  }, [fullScreenCamera])

  const handleLayoutChange = (newLayout: GridLayoutType) => {
    setLayoutType(newLayout)
    setCurrentPage(0)
    setSelectedSlots(new Set())
  }

  const handlePageChange = (page: number) => {
    setCurrentPage(page)
    setSelectedSlots(new Set())
  }

  const handleDragStart = (slotIndex: number) => {
    setDraggedSlot(slotIndex)
    setDraggedCamera(null)
  }

  const handleCameraDragStart = (camera: Camera) => {
    setDraggedCamera(camera)
    setDraggedSlot(null)
  }

  const handleDragOver = (e: React.DragEvent) => {
    e.preventDefault()
  }

  const handleDrop = (targetSlotIndex: number) => {
    const newAssignments = new Map(cameraAssignments)
    const targetGlobalIndex = startIndex + targetSlotIndex

    // Dragging from sidebar
    if (draggedCamera !== null) {
      newAssignments.set(targetGlobalIndex, draggedCamera.id)
      setCameraAssignments(newAssignments)
      setDraggedCamera(null)
      return
    }

    // Dragging between slots
    if (draggedSlot === null || draggedSlot === targetSlotIndex) {
      setDraggedSlot(null)
      return
    }

    const sourceCamera = assignedCameras[draggedSlot]
    const targetCamera = assignedCameras[targetSlotIndex]
    const sourceGlobalIndex = startIndex + draggedSlot

    // Get the original cameras at these positions (before custom assignments)
    const originalSourceCamera = currentPageCameras[draggedSlot]
    const originalTargetCamera = currentPageCameras[targetSlotIndex]

    // Swap cameras
    if (sourceCamera) {
      // Only set assignment if it differs from the original
      if (sourceCamera.id !== originalTargetCamera?.id) {
        newAssignments.set(targetGlobalIndex, sourceCamera.id)
      } else {
        newAssignments.delete(targetGlobalIndex)
      }
    } else {
      newAssignments.delete(targetGlobalIndex)
    }

    if (targetCamera) {
      // Only set assignment if it differs from the original
      if (targetCamera.id !== originalSourceCamera?.id) {
        newAssignments.set(sourceGlobalIndex, targetCamera.id)
      } else {
        newAssignments.delete(sourceGlobalIndex)
      }
    } else {
      newAssignments.delete(sourceGlobalIndex)
    }

    setCameraAssignments(newAssignments)
    setDraggedSlot(null)
  }

  // Full screen camera display
  if (fullScreenCamera !== null) {
    const camera = assignedCameras[fullScreenCamera]
    return (
      <Box position="fixed" top={0} left={0} right={0} bottom={0} bg="black" zIndex={1000}>
        <Box
          position="absolute"
          bottom={4}
          right={4}
          zIndex={10}
          cursor="pointer"
          onClick={handleExitFullScreen}
          color="white"
          fontSize="2xl"
          px={4}
          py={2}
          bg="blackAlpha.700"
          borderRadius="md"
          _hover={{ bg: "blackAlpha.800" }}
          transition="all 0.2s"
        >
          ✕ Exit Full Screen (ESC)
        </Box>
        {camera && (
          <Box h="100vh" position="relative">
            <Flex
              position="absolute"
              top={0}
              left={0}
              right={0}
              justify="space-between"
              align="center"
              px={6}
              py={4}
              bg="blackAlpha.700"
              zIndex={1}
            >
              <Box>
                <Text fontSize="2xl" fontWeight="bold" color="white">
                  {camera.name}
                </Text>
                <Text fontSize="md" color="gray.300">
                  {camera.location}
                </Text>
              </Box>
              <Badge colorPalette={
                camera.status === "recording" || camera.status === "online" ? "green" :
                camera.status === "offline" ? "red" :
                camera.status === "buffering" ? "yellow" : "red"
              }>
                {camera.status}
              </Badge>
            </Flex>
            <Flex
              h="full"
              align="center"
              justify="center"
              direction="column"
              gap={4}
            >
              <FiVideo size={64} color="gray" />
              <Text fontSize="xl" color="gray.400">
                Stream will appear here
              </Text>
            </Flex>
          </Box>
        )}
      </Box>
    )
  }

  return (
    <Flex direction="column" h="100vh" overflow="hidden">
      <StatusBar status={systemStatus} />
      <TopToolbar
        layoutType={layoutType}
        onLayoutChange={handleLayoutChange}
        selectedZone={selectedZone}
        onZoneChange={setSelectedZone}
        selectedPreset={selectedPresetId}
        onPresetChange={handleLoadPreset}
        sequenceConfig={sequenceConfig}
        onSequenceConfigChange={setSequenceConfig}
        zones={zones}
        presets={savedPresets}
        onSavePreset={handleSavePreset}
        onResetView={handleResetView}
      />

      <Flex flex="1" overflow="hidden" position="relative">
        <LeftSidebar isOpen={sidebarOpen} onToggle={() => setSidebarOpen(!sidebarOpen)}>
          <CameraList
            zones={cameraZones}
            onCameraDragStart={handleCameraDragStart}
          />
        </LeftSidebar>

        <Flex direction="column" flex="1" overflow="hidden">
          <Box flex="1" p={4} overflow="hidden">
            <GridLayout
              layoutType={layoutType}
              cameras={assignedCameras}
              selectedSlots={selectedSlots}
              onSlotSelect={handleSlotSelect}
              onSlotDoubleClick={handleSlotDoubleClick}
              onDragStart={handleDragStart}
              onDragOver={handleDragOver}
              onDrop={handleDrop}
            />
          </Box>

          {totalPages > 1 && (
            <PaginationControls
              currentPage={currentPage}
              totalPages={totalPages}
              camerasPerPage={camerasPerPage}
              totalCameras={filteredCameras.length}
              onPageChange={handlePageChange}
            />
          )}
        </Flex>
      </Flex>
    </Flex>
  )
}
