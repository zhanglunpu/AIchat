// API配置
export const API_CONFIG = {
    // API基础URL
    BASE_URL:  'http://localhost:8000', // 替换为您电脑的实际IP地址
    
    // 搜索相关接口
    SEARCH: {
        TEXT: '/search/text',  // 文本搜索接口
        IMAGE: '/search/image' // 图片搜索接口
    },
    
    // 3D模型相关接口
    MODEL: {
        VIEW: '/test3d'    // 3D模型查看接口
    },
    
    AI: {
        QUERY: '/api/ai/query'  // AI问答的API端点
    }
};

/**
 * 获取完整API URL
 * @param {string} endpoint - API端点
 * @returns {string} 完整的API URL
 */
export function getApiUrl(endpoint) {
    return `${API_CONFIG.BASE_URL}${endpoint}`;
}

/**
 * 获取带参数的URL
 * @param {string} endpoint - API端点
 * @param {Object} params - URL参数
 * @returns {string} 带参数的完整URL
 */
export function getUrlWithParams(endpoint, params = {}) {
    const url = getApiUrl(endpoint);
    
    // 不使用URLSearchParams，手动构建查询字符串
    const paramArr = [];
    for (const key in params) {
        if (params.hasOwnProperty(key) && params[key] !== undefined) {
            const value = params[key];
            paramArr.push(`${encodeURIComponent(key)}=${encodeURIComponent(value)}`);
        }
    }
    
    const paramString = paramArr.join('&');
    return paramString ? `${url}?${paramString}` : url;
} 