// 初始化Three.js场景
let scene, camera, renderer;

function initThree() {
    // 创建场景
    scene = new THREE.Scene();
    
    // 创建相机
    camera = new THREE.PerspectiveCamera(75, window.innerWidth / window.innerHeight, 0.1, 1000);
    camera.position.z = 5;
    
    // 创建渲染器
    const container = document.getElementById('three-container');
    renderer = new THREE.WebGLRenderer({ antialias: true });
    renderer.setSize(container.clientWidth, container.clientHeight);
    container.appendChild(renderer.domElement);
    
    // 添加环境光
    const ambientLight = new THREE.AmbientLight(0xffffff, 0.5);
    scene.add(ambientLight);
    
    // 添加方向光
    const directionalLight = new THREE.DirectionalLight(0xffffff, 0.5);
    directionalLight.position.set(0, 1, 0);
    scene.add(directionalLight);
    
    // 开始动画循环
    animate();
}

// 动画循环
function animate() {
    requestAnimationFrame(animate);
    if (scene.children.length > 2) { // 如果场景中有模型(除了两个光源)
        scene.children[2].rotation.y += 0.01; // 旋转模型
    }
    renderer.render(scene, camera);
}

// 加载3D模型
function loadModel(modelUrl) {
    console.log('开始加载3D模型:', modelUrl);
    
    // 清除之前的模型
    while (scene.children.length > 2) { // 保留两个光源
        scene.remove(scene.children[2]);
    }
    
    // 创建GLTFLoader
    const loader = new THREE.GLTFLoader();
    
    loader.load(modelUrl, 
        (gltf) => {
            console.log('模型加载成功:', modelUrl);
            console.log('模型信息:', {
                animations: gltf.animations.length,
                scenes: gltf.scenes.length,
                cameras: gltf.cameras.length,
                asset: gltf.asset
            });
            
            const model = gltf.scene;
            // 自动调整模型大小和位置
            const box = new THREE.Box3().setFromObject(model);
            const center = box.getCenter(new THREE.Vector3());
            const size = box.getSize(new THREE.Vector3());
            
            const maxDim = Math.max(size.x, size.y, size.z);
            const scale = 3 / maxDim;
            model.scale.setScalar(scale);
            
            model.position.sub(center.multiplyScalar(scale));
            
            scene.add(model);
            console.log('模型已添加到场景');
        },
        (progress) => {
            console.log('加载进度:', {
                total: progress.total,
                loaded: progress.loaded,
                percent: (progress.loaded / progress.total * 100).toFixed(2) + '%'
            });
        },
        (error) => {
            console.error('模型加载失败:', modelUrl, error);
        }
    );
}

// 监听3D模型查看器的显示
document.addEventListener('htmx:afterSwap', (event) => {
    if (event.detail.target.id === 'model-viewer' && !event.detail.target.classList.contains('hidden')) {
        if (!renderer) {
            initThree();
        }
        // 假设模型URL从后端返回并存储在data-model-url属性中
        const modelUrl = event.detail.target.getAttribute('data-model-url');
        if (modelUrl) {
            loadModel(modelUrl);
        }
    }
}); 