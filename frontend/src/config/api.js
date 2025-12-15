// API configuration
const API_BASE_URL = import.meta.env.VITE_API_URL || "http://localhost:8000";

export const apiClient = {
  baseURL: API_BASE_URL,

  get: async (endpoint, config = {}) => {
    const axios = (await import("axios")).default;
    return axios.get(`${API_BASE_URL}${endpoint}`, config);
  },

  post: async (endpoint, data, config = {}) => {
    const axios = (await import("axios")).default;
    return axios.post(`${API_BASE_URL}${endpoint}`, data, config);
  },
};

export default API_BASE_URL;
