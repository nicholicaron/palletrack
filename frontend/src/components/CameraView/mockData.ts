import type { Camera, CameraZone, CameraStatus } from "./types"

const zones = ["Loading Dock", "Warehouse Floor", "Quality Control", "Shipping Area", "Outdoor"]
const locations = [
  "Bay A-1", "Bay A-2", "Bay A-3", "Bay B-1", "Bay B-2", "Bay B-3",
  "Dock 1", "Dock 2", "Dock 3", "Dock 4",
  "QC Station 1", "QC Station 2", "QC Station 3",
  "Entrance", "Exit", "Parking Lot",
  "Aisle 1", "Aisle 2", "Aisle 3", "Aisle 4", "Aisle 5",
  "Office Entrance", "Breakroom", "Storage Room"
]

function getRandomStatus(): CameraStatus {
  const rand = Math.random()
  if (rand < 0.5) return "recording" // 50% recording
  if (rand < 0.75) return "online" // 25% online
  if (rand < 0.85) return "buffering" // 10% buffering
  if (rand < 0.95) return "offline" // 10% offline
  return "error" // 5% error
}

function generateMockCameras(count: number = 64): Camera[] {
  const cameras: Camera[] = []

  for (let i = 0; i < count; i++) {
    const zone = zones[i % zones.length]
    const location = locations[i % locations.length]
    const status = getRandomStatus()

    cameras.push({
      id: `cam-${i + 1}`,
      name: `Camera ${i + 1}`,
      location,
      zone,
      status,
      ipAddress: `192.168.1.${100 + i}`,
      resolution: i % 3 === 0 ? "4K" : i % 2 === 0 ? "1080p" : "720p",
      fps: i % 2 === 0 ? 30 : 60,
      bitrate: `${Math.floor(Math.random() * 5 + 2)}Mbps`,
      supportsPTZ: i % 4 === 0,
      lastMotionDetected: status === "recording" ? new Date(Date.now() - Math.random() * 3600000) : undefined,
    })
  }

  return cameras
}

export function generateMockCameraZones(): CameraZone[] {
  const cameras = generateMockCameras(64)

  return zones.map((zoneName, index) => ({
    id: `zone-${index + 1}`,
    name: zoneName,
    cameras: cameras.filter((cam) => cam.zone === zoneName),
  }))
}

export function generateMockCameras64(): Camera[] {
  return generateMockCameras(64)
}
