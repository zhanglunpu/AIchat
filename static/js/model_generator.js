import * as THREE from 'three';
import { GLTFExporter } from 'three/examples/jsm/exporters/GLTFExporter.js';
import fs from 'fs';
import path from 'path';

// 创建基础场景
const scene = new THREE.Scene();

// 创建不同形状的模型生成函数
const modelGenerators = {
    // 金银花 - 使用多个小球体组成花朵形状
    jinyinhua: () => {
        const group = new THREE.Group();
        const flowerGeometry = new THREE.SphereGeometry(0.3, 32, 32);
        const flowerMaterial = new THREE.MeshStandardMaterial({ 
            color: 0xFFFFE0,
            roughness: 0.5,
            metalness: 0.2
        });

        // 创建花瓣
        for (let i = 0; i < 5; i++) {
            const petal = new THREE.Mesh(flowerGeometry, flowerMaterial);
            petal.position.x = Math.cos(i * Math.PI * 0.4) * 0.5;
            petal.position.y = Math.sin(i * Math.PI * 0.4) * 0.5;
            group.add(petal);
        }
        return group;
    },

    // 槐花 - 使用圆锥体和球体组合
    huaihua: () => {
        const group = new THREE.Group();
        const petalGeometry = new THREE.ConeGeometry(0.2, 0.4, 32);
        const centerGeometry = new THREE.SphereGeometry(0.15, 32, 32);
        const material = new THREE.MeshStandardMaterial({ 
            color: 0xFFFFFF,
            roughness: 0.6,
            metalness: 0.1
        });

        // 创建花瓣
        for (let i = 0; i < 6; i++) {
            const petal = new THREE.Mesh(petalGeometry, material);
            petal.position.x = Math.cos(i * Math.PI / 3) * 0.3;
            petal.position.z = Math.sin(i * Math.PI / 3) * 0.3;
            petal.rotation.x = Math.PI * 0.15;
            petal.rotation.y = i * Math.PI / 3;
            group.add(petal);
        }

        // 添加花心
        const center = new THREE.Mesh(centerGeometry, material);
        group.add(center);
        return group;
    },

    // 枸杞 - 使用小球体表示果实
    gouqi: () => {
        const group = new THREE.Group();
        const berryGeometry = new THREE.SphereGeometry(0.2, 32, 32);
        const berryMaterial = new THREE.MeshStandardMaterial({ 
            color: 0xFF4500,
            roughness: 0.3,
            metalness: 0.1
        });

        // 创建多个果实
        for (let i = 0; i < 8; i++) {
            const berry = new THREE.Mesh(berryGeometry, berryMaterial);
            berry.position.x = (Math.random() - 0.5) * 0.8;
            berry.position.y = (Math.random() - 0.5) * 0.8;
            berry.position.z = (Math.random() - 0.5) * 0.8;
            berry.scale.set(
                0.8 + Math.random() * 0.4,
                0.8 + Math.random() * 0.4,
                0.8 + Math.random() * 0.4
            );
            group.add(berry);
        }
        return group;
    },

    // 党参 - 使用圆柱体表示根部
    dangshen: () => {
        const group = new THREE.Group();
        const rootGeometry = new THREE.CylinderGeometry(0.2, 0.1, 1.5, 32);
        const rootMaterial = new THREE.MeshStandardMaterial({ 
            color: 0xDEB887,
            roughness: 0.8,
            metalness: 0.1
        });

        const root = new THREE.Mesh(rootGeometry, rootMaterial);
        root.rotation.x = Math.PI * 0.1;
        group.add(root);

        // 添加一些纹理细节
        const detailGeometry = new THREE.CylinderGeometry(0.02, 0.02, 0.2, 8);
        for (let i = 0; i < 12; i++) {
            const detail = new THREE.Mesh(detailGeometry, rootMaterial);
            detail.position.y = (Math.random() - 0.5) * 1.2;
            detail.position.x = (Math.random() - 0.5) * 0.3;
            detail.position.z = (Math.random() - 0.5) * 0.3;
            detail.rotation.x = Math.random() * Math.PI;
            detail.rotation.z = Math.random() * Math.PI;
            group.add(detail);
        }
        return group;
    },

    // 百合 - 使用多个曲面创建花瓣
    baihe: () => {
        const group = new THREE.Group();
        const petalShape = new THREE.Shape();
        petalShape.moveTo(0, 0);
        petalShape.quadraticCurveTo(0.3, 0.5, 0, 1);
        petalShape.quadraticCurveTo(-0.3, 0.5, 0, 0);

        const petalGeometry = new THREE.ShapeGeometry(petalShape);
        const petalMaterial = new THREE.MeshStandardMaterial({ 
            color: 0xFFFAFA,
            side: THREE.DoubleSide,
            roughness: 0.4,
            metalness: 0.2
        });

        // 创建6片花瓣
        for (let i = 0; i < 6; i++) {
            const petal = new THREE.Mesh(petalGeometry, petalMaterial);
            petal.rotation.y = i * Math.PI / 3;
            petal.rotation.x = Math.PI * 0.2;
            group.add(petal);
        }
        return group;
    }
};

// 导出模型
async function exportModel(model, filename) {
    const exporter = new GLTFExporter();
    const gltf = await new Promise((resolve) => {
        exporter.parse(model, resolve, { binary: true });
    });

    const buffer = Buffer.from(gltf);
    const filepath = path.join(__dirname, '..', 'static', 'models', filename);
    fs.writeFileSync(filepath, buffer);
}

// 生成所有模型
async function generateAllModels() {
    // 添加灯光
    const ambientLight = new THREE.AmbientLight(0xFFFFFF, 0.5);
    const directionalLight = new THREE.DirectionalLight(0xFFFFFF, 0.8);
    directionalLight.position.set(0, 1, 0);
    scene.add(ambientLight);
    scene.add(directionalLight);

    // 生成并导出每个模型
    for (const [name, generator] of Object.entries(modelGenerators)) {
        scene.add(generator());
        await exportModel(scene, `${name}.glb`);
        scene.remove(scene.children[scene.children.length - 1]);
    }
}

generateAllModels().catch(console.error); 