import React, { useState } from 'react'

function Tooltip({ content, children }) {
  const [isVisible, setIsVisible] = useState(false)

  return (
    <div 
      className="relative inline-block"
      onMouseEnter={() => setIsVisible(true)}
      onMouseLeave={() => setIsVisible(false)}
    >
      {children}
      {isVisible && (
        <div className="absolute z-50 w-64 p-3 text-sm text-white bg-gray-900 rounded-lg shadow-xl bottom-full left-1/2 transform -translate-x-1/2 mb-2 animate-fade-in">
          <div className="absolute bottom-0 left-1/2 transform -translate-x-1/2 translate-y-full">
            <div className="border-4 border-transparent border-t-gray-900"></div>
          </div>
          {content}
        </div>
      )}
    </div>
  )
}

export default Tooltip

