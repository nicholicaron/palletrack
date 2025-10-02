import { Box, IconButton } from "@chakra-ui/react"
import { FiChevronLeft, FiChevronRight } from "react-icons/fi"

interface LeftSidebarProps {
  isOpen: boolean
  onToggle: () => void
  children: React.ReactNode
}

function LeftSidebar({ isOpen, onToggle, children }: LeftSidebarProps) {
  return (
    <>
      {/* Sidebar */}
      <Box
        position="relative"
        w={isOpen ? "300px" : "0"}
        transition="width 0.3s"
        overflow="hidden"
        borderRight={isOpen ? "1px solid" : "none"}
        borderColor="border"
        bg="bg.subtle"
      >
        {isOpen && (
          <Box h="full" p={4} overflowY="auto">
            {children}
          </Box>
        )}
      </Box>

      {/* Toggle Button */}
      <Box
        position="absolute"
        left={isOpen ? "300px" : "0"}
        top="50%"
        transform="translateY(-50%)"
        transition="left 0.3s"
        zIndex={10}
      >
        <IconButton
          size="sm"
          variant="solid"
          onClick={onToggle}
          aria-label={isOpen ? "Close sidebar" : "Open sidebar"}
        >
          {isOpen ? <FiChevronLeft /> : <FiChevronRight />}
        </IconButton>
      </Box>
    </>
  )
}

export default LeftSidebar
