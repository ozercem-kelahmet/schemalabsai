import React from 'react'

interface Props {
  content: string
}

export function MessageContent({ content }: Props) {
  // Split content into parts (text and tables)
  const parts = content.split(/(\n\|[^\n]+\|[\s\S]*?\n(?=\n|$))/g)
  
  return (
    <div className="space-y-3">
      {parts.map((part, idx) => {
        // Check if this part is a table
        if (part.includes('|') && part.split('\n').filter(l => l.includes('|')).length >= 2) {
          return <TableRenderer key={idx} tableText={part} />
        }
        // Regular text
        if (part.trim()) {
          return (
            <div key={idx} className="whitespace-pre-wrap">
              {part.split('\n').map((line, i) => (
                <p key={i} className={line.trim() ? 'mb-2' : 'mb-1'}>{line}</p>
              ))}
            </div>
          )
        }
        return null
      })}
    </div>
  )
}

function TableRenderer({ tableText }: { tableText: string }) {
  const lines = tableText.trim().split('\n').filter(l => l.includes('|'))
  
  if (lines.length < 2) return <pre>{tableText}</pre>
  
  const parseRow = (line: string) => {
    return line.split('|').map(cell => cell.trim()).filter(cell => cell && !cell.match(/^-+$/))
  }
  
  const headers = parseRow(lines[0])
  const dataRows = lines.slice(1)
    .filter(line => !line.match(/^\|?[\s-|]+\|?$/)) // Skip separator rows
    .map(parseRow)
  
  return (
    <div className="overflow-x-auto my-3">
      <table className="min-w-full border border-gray-200 rounded-lg overflow-hidden">
        <thead className="bg-gray-50">
          <tr>
            {headers.map((header, i) => (
              <th key={i} className="px-4 py-2 text-left text-sm font-semibold text-gray-700 border-b">
                {header}
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {dataRows.map((row, i) => (
            <tr key={i} className={i % 2 === 0 ? 'bg-white' : 'bg-gray-50'}>
              {row.map((cell, j) => (
                <td key={j} className="px-4 py-2 text-sm text-gray-600 border-b">
                  {cell}
                </td>
              ))}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  )
}

export default MessageContent
