export function SchemaIcon({
  className = "w-5 h-5",
  variant = "default",
}: {
  className?: string
  variant?: "default" | "inverted"
}) {
  const strokeColor = variant === "inverted" ? "black" : "white"
  const fillColor = variant === "inverted" ? "black" : "white"

  return (
    <svg viewBox="0 0 70 70" fill="none" xmlns="http://www.w3.org/2000/svg" className={className}>
      {/* Left Bracket [ */}
      <path d="M20 0H0V70H20" stroke={strokeColor} strokeWidth="8" />
      {/* Right Bracket ] */}
      <path d="M50 70H70V0H50" stroke={strokeColor} strokeWidth="8" />
      {/* Central Data Point / Cursor block */}
      <rect x="28" y="28" width="14" height="14" fill={fillColor} />
    </svg>
  )
}
