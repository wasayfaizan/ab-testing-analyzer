import React from 'react'

function SectionHeader({ icon, title, subtitle }) {
  return (
    <div className="mb-6">
      <div className="flex items-center gap-3 mb-2">
        {icon && (
          <div className="w-8 h-8 flex items-center justify-center text-gray-600">
            {icon}
          </div>
        )}
        <h2 className="text-xl font-bold text-gray-900">{title}</h2>
      </div>
      {subtitle && (
        <p className="text-sm text-gray-600 ml-11">{subtitle}</p>
      )}
      <div className="mt-3 h-0.5 bg-gradient-to-r from-blue-500 via-indigo-500 to-purple-500 rounded-full"></div>
    </div>
  )
}

export default SectionHeader

