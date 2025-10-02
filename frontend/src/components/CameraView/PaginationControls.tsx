import { Flex, IconButton, Button } from "@chakra-ui/react"
import { FiChevronLeft, FiChevronRight } from "react-icons/fi"
import {
  MenuContent,
  MenuItem,
  MenuRoot,
  MenuTrigger,
} from "../ui/menu"

interface PaginationControlsProps {
  currentPage: number
  totalPages: number
  camerasPerPage: number
  totalCameras: number
  onPageChange: (page: number) => void
}

function PaginationControls({
  currentPage,
  totalPages,
  onPageChange,
}: PaginationControlsProps) {
  if (totalPages <= 1) return null

  return (
    <Flex
      justify="center"
      align="center"
      gap={4}
      py={3}
      px={4}
      borderTop="1px solid"
      borderColor="border"
      bg="bg.subtle"
    >
      <IconButton
        size="sm"
        variant="outline"
        onClick={() => onPageChange(currentPage - 1)}
        disabled={currentPage === 0}
        aria-label="Previous page"
      >
        <FiChevronLeft />
      </IconButton>

      <MenuRoot>
        <MenuTrigger asChild>
          <Button variant="outline" size="sm">
            Page {currentPage + 1} of {totalPages}
          </Button>
        </MenuTrigger>
        <MenuContent>
          {Array.from({ length: totalPages }).map((_, index) => (
            <MenuItem
              key={index}
              value={String(index)}
              onClick={() => onPageChange(index)}
            >
              Page {index + 1} {index === currentPage && "✓"}
            </MenuItem>
          ))}
        </MenuContent>
      </MenuRoot>

      <IconButton
        size="sm"
        variant="outline"
        onClick={() => onPageChange(currentPage + 1)}
        disabled={currentPage >= totalPages - 1}
        aria-label="Next page"
      >
        <FiChevronRight />
      </IconButton>
    </Flex>
  )
}

export default PaginationControls
