import React from 'react';
import { BrainCog } from 'lucide-react';

const Header: React.FC = () => {
  return (
    <header className="bg-gradient-to-r from-indigo-600 via-blue-600 to-sky-600 shadow-lg py-6 gradient-animate relative z-20">
      <div className="absolute inset-0 bg-[radial-gradient(circle_at_center,_var(--tw-gradient-stops))] from-white/10 via-transparent to-transparent"></div>
      <div className="container mx-auto px-4 max-w-3xl flex items-center relative">
        <div className="relative bg-white/10 p-2 rounded-lg mr-4 animate-float backdrop-blur-sm">
          <div className="absolute inset-0 bg-white/20 rounded-lg pulse-ring"></div>
          <BrainCog className="h-8 w-8 text-white relative z-10" />
        </div>
        <div className="slide-up">
          <h1 className="text-2xl font-bold text-white">
            Mental Health Counselor Guidance
          </h1>
          <p className="text-blue-100 text-sm mt-1">
            AI-powered assistance for mental health professionals
          </p>
        </div>
      </div>
    </header>
  );
};

export default Header;