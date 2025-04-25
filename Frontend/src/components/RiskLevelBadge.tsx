import React from 'react';

interface RiskLevelBadgeProps {
  level: string;
}

const RiskLevelBadge: React.FC<RiskLevelBadgeProps> = ({ level }) => {
  const getLevelStyles = () => {
    switch (level.toLowerCase()) {
      case 'high':
        return 'bg-red-100 text-red-800 border-red-200 ring-red-500';
      case 'moderate':
        return 'bg-orange-100 text-orange-800 border-orange-200 ring-orange-500';
      case 'low':
        return 'bg-green-100 text-green-800 border-green-200 ring-green-500';
      default:
        return 'bg-gray-100 text-gray-800 border-gray-200 ring-gray-500';
    }
  };

  return (
    <span className={`inline-flex items-center px-3 py-1 rounded-full text-sm font-medium border ring-1 ring-opacity-20 ${getLevelStyles()}`}>
      {level}
    </span>
  );
};

export default RiskLevelBadge;