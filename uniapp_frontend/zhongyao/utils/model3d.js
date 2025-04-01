import { API_CONFIG, getUrlWithParams } from '../config/api.js';

// 3D模型基础URL
const BASE_URL = `${API_CONFIG.BASE_URL}${API_CONFIG.MODEL.VIEW}`;

/**
 * 构造3D模型的完整URL
 * @param {Object|string} herb - 药材对象或ID
 * @returns {string} 完整的3D模型URL
 */
export function build3DModelUrl(herb) {
    console.log('构造3D模型URL, 传入参数:', JSON.stringify(herb, null, 2));
    
    // 如果只传入了ID,则只返回基本URL
    if (typeof herb === 'string' || typeof herb === 'number') {
        const modelPath = `/static/images/3d/${encodeURIComponent(herb)}.glb`;
        const url = `${BASE_URL}?model=${modelPath}`;
        console.log('返回基础URL:', url);
        return url;
    }
    
    // 构造模型路径
    const modelPath = `/static/images/3d/${encodeURIComponent(herb.model)}`;
    
    // 构造完整URL,包含所有参数
    const url = `${BASE_URL}?model=${modelPath}&name=${encodeURIComponent(herb.name || '-')}&pinyin=${encodeURIComponent(herb.pinyin || '-')}&taste=${encodeURIComponent(herb.taste || '-')}&meridian=${encodeURIComponent(herb.meridian || '-')}&effect=${encodeURIComponent(herb.effect || '-')}`;
    
    console.log('返回完整URL:', url);
    return url;
}

/**
 * 打开3D模型查看器
 * @param {Object} herb - 药材对象
 * @returns {Promise} 导航结果
 */
export function view3DModel(herb) {
    return new Promise((resolve, reject) => {
        try {
            console.log('准备打开3D模型,药材数据:', JSON.stringify(herb, null, 2));
            
            // 检查model_url是否有效
            if (!herb.model_url) {
                throw new Error('没有找到药材的3D模型');
            }
            
            // 构建3D模型查看页面的URL
            const params = {
                model: herb.model_url,
                name: herb.name || '未知药材',
                pinyin: herb.pinyin || '-',
                taste: herb.taste || '-',
                meridian: herb.meridian || '-',
                effect: herb.effect || '-'
            };
            
            console.log('构建3D模型URL参数:', params);
            
            const modelViewUrl = getUrlWithParams(API_CONFIG.MODEL.VIEW, params);
            console.log('完整的3D模型查看URL:', modelViewUrl);

            // 打开WebView页面
            uni.navigateTo({
                url: `/pages/webview/webview?url=${encodeURIComponent(modelViewUrl)}&title=${encodeURIComponent(herb.name)}`,
                success: () => {
                    console.log('成功打开3D模型页面');
                    resolve();
                },
                fail: (err) => {
                    console.error('打开3D模型页面失败:', err);
                    reject(err);
                }
            });
        } catch (err) {
            console.error('view3DModel执行失败:', err);
            // 显示友好的错误提示
            uni.showToast({
                title: err.message || '打开3D模型失败',
                icon: 'none'
            });
            reject(err);
        }
    });
}

/**
 * 处理API返回的药材数据,提取3D模型所需信息
 * @param {Object} herb - API返回的原始数据
 * @returns {Object} 处理后的药材信息
 */
export function processHerbData(herb) {
    console.log('API返回的原始数据:', JSON.stringify(herb, null, 2));
    
    // 从properties中提取数据
    const properties = herb.properties || {};
    const taste = properties.taste ? `${properties.nature || ''}${properties.taste || ''}` : '-';
    const meridians = properties.meridians || [];
    const functions = properties.functions || [];
    
    // 构造model_url - 使用与后端一致的静态资源路径
    const modelUrl = herb.model_file ? `/static/images/3d/${herb.model_file}` : '';
    
    console.log('构造的模型URL:', modelUrl);
    
    return {
        name: herb.name || '未知药材',
        pinyin: herb.pinyin || '-',
        taste: taste,
        meridian: meridians.join('、') || '-',
        effect: functions.join('、') || '-',
        model_url: modelUrl
    };
} 