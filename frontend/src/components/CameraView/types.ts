// Camera status types
export type CameraStatus = "online" | "offline" | "recording" | "buffering" | "error"

// Grid layout types
export type GridLayoutType = "1x1" | "2x2" | "3x3" | "4x4" | "1+5" | "1+7" | "2x3"

// Camera interface
export interface Camera {
  id: string
  name: string
  location: string
  zone: string
  status: CameraStatus
  ipAddress?: string
  resolution?: string
  fps?: number
  bitrate?: string
  supportsPTZ?: boolean
  lastMotionDetected?: Date
  thumbnailUrl?: string
}

// Camera slot in the grid
export interface CameraSlot {
  slotIndex: number
  camera: Camera | null
  isSelected: boolean
  isExpanded: boolean
}

// Grid layout configuration
export interface GridLayoutConfig {
  type: GridLayoutType
  columns: number
  rows: number
  slots: number
  specialLayout?: boolean // For 1+5 and 1+7 layouts
}

// View preset
export interface ViewPreset {
  id: string
  name: string
  layoutType: GridLayoutType
  cameraAssignments: Map<number, string> // slotIndex -> cameraId
  createdAt: Date
  isSystemPreset: boolean
}

// Sequence mode configuration
export interface SequenceConfig {
  enabled: boolean
  dwellTime: number // seconds
  mode: "all-cameras" | "camera-group" | "presets"
  targetGroupId?: string
  pauseOnMotion: boolean
}

// User preferences
export interface UserPreferences {
  lastLayoutType: GridLayoutType
  autoplay: boolean
  defaultZone?: string
  savedPresets: ViewPreset[]
  sequenceConfig: SequenceConfig
}

// Camera context menu action
export interface ContextMenuAction {
  label: string
  action: (camera: Camera, slotIndex: number) => void
  icon?: string
  disabled?: boolean
}

// Pagination state
export interface PaginationState {
  currentPage: number
  totalPages: number
  camerasPerPage: number
  startIndex: number
  endIndex: number
}

// Multi-selection state
export interface SelectionState {
  selectedSlots: Set<number>
  lastSelectedIndex: number | null
  anchorIndex: number | null
}

// Zone/location grouping
export interface CameraZone {
  id: string
  name: string
  cameras: Camera[]
}

// System status
export interface SystemStatus {
  totalCameras: number
  onlineCameras: number
  offlineCameras: number
  recordingCameras: number
  totalBandwidth: string
  storageRemaining: string
  activeAlerts: number
}
