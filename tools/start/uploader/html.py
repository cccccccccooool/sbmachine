# -*- coding: utf-8 -*-

html_content = """<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>CNB 工作区文件上传器</title>
    <link href="https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700&family=Noto+Sans+SC:wght@300;400;500;700&display=swap" rel="stylesheet">
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css">
    <style>
        :root {
            --bg-color: #0c0813;
            --panel-bg: rgba(22, 16, 35, 0.65);
            --panel-border: rgba(255, 255, 255, 0.08);
            --text-primary: #f3f1f6;
            --text-secondary: #a49db2;
            --primary-gradient: linear-gradient(135deg, #a855f7 0%, #06b6d4 100%);
            --accent-cyan: #06b6d4;
            --accent-purple: #a855f7;
            --accent-green: #10b981;
            --accent-red: #ef4444;
            --hover-bg: rgba(255, 255, 255, 0.05);
            --transition-speed: 0.25s;
        }

        * {
            box-sizing: border-box;
            margin: 0;
            padding: 0;
        }

        body {
            font-family: 'Outfit', 'Noto Sans SC', sans-serif;
            background-color: var(--bg-color);
            background-image: 
                radial-gradient(at 10% 10%, rgba(168, 85, 247, 0.15) 0px, transparent 50%),
                radial-gradient(at 90% 90%, rgba(6, 182, 212, 0.15) 0px, transparent 50%);
            background-attachment: fixed;
            color: var(--text-primary);
            min-height: 100vh;
            display: flex;
            flex-direction: column;
            align-items: center;
            padding: 2rem 1rem;
        }

        .container {
            width: 100%;
            max-width: 1200px;
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 2rem;
            margin-top: 1.5rem;
        }

        @media (max-width: 968px) {
            .container {
                grid-template-columns: 1fr;
            }
        }

        header {
            width: 100%;
            max-width: 1200px;
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 1rem;
        }

        .logo {
            display: flex;
            align-items: center;
            gap: 0.75rem;
        }

        .logo i {
            font-size: 2rem;
            background: var(--primary-gradient);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }

        .logo h1 {
            font-size: 1.75rem;
            font-weight: 700;
            letter-spacing: -0.5px;
        }

        .logo span {
            font-size: 0.8rem;
            background: rgba(168, 85, 247, 0.2);
            color: var(--accent-purple);
            padding: 0.2rem 0.6rem;
            border-radius: 99px;
            border: 1px solid rgba(168, 85, 247, 0.3);
            font-weight: 500;
        }

        .glass-panel {
            background: var(--panel-bg);
            backdrop-filter: blur(20px);
            -webkit-backdrop-filter: blur(20px);
            border: 1px solid var(--panel-border);
            border-radius: 16px;
            padding: 1.5rem;
            box-shadow: 0 10px 30px rgba(0, 0, 0, 0.5);
            display: flex;
            flex-direction: column;
        }

        /* Uploader Area */
        .dropzone {
            border: 2px dashed rgba(255, 255, 255, 0.15);
            border-radius: 12px;
            padding: 3rem 2rem;
            text-align: center;
            cursor: pointer;
            transition: all var(--transition-speed) ease;
            background: rgba(255, 255, 255, 0.01);
            position: relative;
            overflow: hidden;
            display: flex;
            flex-direction: column;
            align-items: center;
            justify-content: center;
            gap: 1rem;
        }

        .dropzone:hover, .dropzone.dragover {
            border-color: var(--accent-cyan);
            background: rgba(6, 182, 212, 0.03);
            box-shadow: 0 0 20px rgba(6, 182, 212, 0.1);
        }

        .dropzone.dragover {
            transform: scale(0.99);
        }

        .dropzone-icon {
            font-size: 3.5rem;
            color: var(--text-secondary);
            transition: transform var(--transition-speed) ease;
        }

        .dropzone:hover .dropzone-icon {
            transform: translateY(-5px);
            color: var(--accent-cyan);
        }

        .dropzone-text h3 {
            font-size: 1.2rem;
            font-weight: 600;
            margin-bottom: 0.5rem;
        }

        .dropzone-text p {
            color: var(--text-secondary);
            font-size: 0.9rem;
        }

        .upload-options {
            display: flex;
            gap: 1rem;
            margin-top: 1rem;
            z-index: 10;
        }

        .btn {
            padding: 0.6rem 1.2rem;
            border-radius: 8px;
            font-family: inherit;
            font-weight: 500;
            font-size: 0.9rem;
            cursor: pointer;
            transition: all var(--transition-speed) ease;
            display: inline-flex;
            align-items: center;
            gap: 0.5rem;
            border: none;
        }

        .btn-primary {
            background: var(--primary-gradient);
            color: white;
            box-shadow: 0 4px 15px rgba(168, 85, 247, 0.3);
        }

        .btn-primary:hover {
            opacity: 0.9;
            transform: translateY(-1px);
        }

        .btn-secondary {
            background: rgba(255, 255, 255, 0.08);
            color: var(--text-primary);
            border: 1px solid rgba(255, 255, 255, 0.08);
        }

        .btn-secondary:hover {
            background: var(--hover-bg);
            border-color: rgba(255, 255, 255, 0.2);
        }

        .btn-danger {
            background: rgba(239, 68, 68, 0.2);
            color: #fca5a5;
            border: 1px solid rgba(239, 68, 68, 0.3);
        }

        .btn-danger:hover {
            background: rgba(239, 68, 68, 0.3);
        }

        .btn-sm {
            padding: 0.35rem 0.75rem;
            font-size: 0.8rem;
            border-radius: 6px;
        }

        /* File Upload Queue */
        .queue-container {
            margin-top: 1.5rem;
            flex-grow: 1;
            display: flex;
            flex-direction: column;
            min-height: 250px;
        }

        .panel-title {
            font-size: 1.1rem;
            font-weight: 600;
            margin-bottom: 1rem;
            display: flex;
            justify-content: space-between;
            align-items: center;
        }

        .queue-stats {
            background: rgba(255, 255, 255, 0.02);
            border: 1px solid var(--panel-border);
            border-radius: 8px;
            padding: 0.75rem 1rem;
            margin-bottom: 1rem;
            display: grid;
            grid-template-columns: repeat(4, 1fr);
            gap: 1rem;
            font-size: 0.85rem;
        }

        @media (max-width: 480px) {
            .queue-stats {
                grid-template-columns: 1fr 1fr;
            }
        }

        .stat-item {
            display: flex;
            flex-direction: column;
            gap: 0.25rem;
        }

        .stat-label {
            color: var(--text-secondary);
            font-size: 0.75rem;
        }

        .stat-val {
            font-weight: 600;
        }

        .progress-bar-container {
            width: 100%;
            height: 6px;
            background: rgba(255, 255, 255, 0.05);
            border-radius: 99px;
            overflow: hidden;
            margin-bottom: 1.5rem;
        }

        .progress-bar {
            height: 100%;
            width: 0%;
            background: var(--primary-gradient);
            border-radius: 99px;
            transition: width 0.2s ease;
        }

        .queue-list {
            list-style: none;
            overflow-y: auto;
            max-height: 280px;
            display: flex;
            flex-direction: column;
            gap: 0.5rem;
            padding-right: 0.25rem;
            flex-grow: 1;
        }

        .queue-list::-webkit-scrollbar {
            width: 6px;
        }
        .queue-list::-webkit-scrollbar-track {
            background: transparent;
        }
        .queue-list::-webkit-scrollbar-thumb {
            background: rgba(255, 255, 255, 0.1);
            border-radius: 99px;
        }
        .queue-list::-webkit-scrollbar-thumb:hover {
            background: rgba(255, 255, 255, 0.2);
        }

        .queue-item {
            background: rgba(255, 255, 255, 0.02);
            border: 1px solid var(--panel-border);
            border-radius: 8px;
            padding: 0.75rem 1rem;
            display: flex;
            flex-direction: column;
            gap: 0.5rem;
        }

        .queue-item-header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            font-size: 0.85rem;
        }

        .queue-item-name {
            font-weight: 500;
            white-space: nowrap;
            overflow: hidden;
            text-overflow: ellipsis;
            max-width: 70%;
        }

        .queue-item-meta {
            color: var(--text-secondary);
            font-size: 0.75rem;
            display: flex;
            align-items: center;
            gap: 0.5rem;
        }

        .queue-item-status {
            font-weight: 600;
            font-size: 0.75rem;
            padding: 0.1rem 0.4rem;
            border-radius: 4px;
        }

        .status-pending { background: rgba(255, 255, 255, 0.08); color: var(--text-secondary); }
        .status-uploading { background: rgba(6, 182, 212, 0.15); color: var(--accent-cyan); }
        .status-success { background: rgba(16, 185, 129, 0.15); color: var(--accent-green); }
        .status-error { background: rgba(239, 68, 68, 0.15); color: var(--accent-red); }

        .queue-item-progress-bg {
            width: 100%;
            height: 4px;
            background: rgba(255, 255, 255, 0.03);
            border-radius: 99px;
            overflow: hidden;
        }

        .queue-item-progress {
            height: 100%;
            width: 0%;
            background: var(--accent-cyan);
            border-radius: 99px;
            transition: width 0.15s ease;
        }

        /* File Explorer Panel */
        .explorer-header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 1rem;
            gap: 1rem;
        }

        .breadcrumbs {
            display: flex;
            align-items: center;
            gap: 0.4rem;
            font-size: 0.9rem;
            font-weight: 500;
            color: var(--text-secondary);
            overflow-x: auto;
            white-space: nowrap;
            padding-bottom: 0.25rem;
            flex-grow: 1;
        }

        .breadcrumbs::-webkit-scrollbar {
            height: 4px;
        }
        .breadcrumbs::-webkit-scrollbar-thumb {
            background: rgba(255, 255, 255, 0.1);
            border-radius: 99px;
        }

        .breadcrumb-item {
            cursor: pointer;
            transition: color var(--transition-speed);
        }

        .breadcrumb-item:hover {
            color: var(--text-primary);
        }

        .breadcrumb-separator {
            color: rgba(255, 255, 255, 0.15);
        }

        .explorer-search {
            position: relative;
            margin-bottom: 1rem;
        }

        .explorer-search input {
            width: 100%;
            background: rgba(0, 0, 0, 0.2);
            border: 1px solid var(--panel-border);
            border-radius: 8px;
            padding: 0.6rem 1rem 0.6rem 2.2rem;
            color: var(--text-primary);
            font-family: inherit;
            font-size: 0.9rem;
            outline: none;
            transition: border-color var(--transition-speed);
        }

        .explorer-search input:focus {
            border-color: var(--accent-purple);
        }

        .explorer-search i {
            position: absolute;
            left: 0.8rem;
            top: 50%;
            transform: translateY(-50%);
            color: var(--text-secondary);
            font-size: 0.9rem;
        }

        .explorer-actions {
            display: flex;
            gap: 0.5rem;
        }

        .file-list-container {
            border: 1px solid var(--panel-border);
            border-radius: 10px;
            background: rgba(0, 0, 0, 0.15);
            overflow: hidden;
            flex-grow: 1;
            display: flex;
            flex-direction: column;
            min-height: 450px;
        }

        .file-list-header {
            display: grid;
            grid-template-columns: auto 1fr auto auto;
            gap: 1rem;
            padding: 0.75rem 1rem;
            background: rgba(255, 255, 255, 0.02);
            border-bottom: 1px solid var(--panel-border);
            font-size: 0.8rem;
            font-weight: 600;
            color: var(--text-secondary);
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }

        .file-list {
            list-style: none;
            overflow-y: auto;
            max-height: 400px;
            flex-grow: 1;
        }

        .file-list::-webkit-scrollbar {
            width: 6px;
        }
        .file-list::-webkit-scrollbar-track {
            background: transparent;
        }
        .file-list::-webkit-scrollbar-thumb {
            background: rgba(255, 255, 255, 0.1);
            border-radius: 99px;
        }

        .file-item {
            display: grid;
            grid-template-columns: auto 1fr auto auto;
            align-items: center;
            gap: 1rem;
            padding: 0.75rem 1rem;
            border-bottom: 1px solid rgba(255, 255, 255, 0.03);
            font-size: 0.9rem;
            transition: background-color var(--transition-speed);
        }

        .file-item:last-child {
            border-bottom: none;
        }

        .file-item:hover {
            background-color: var(--hover-bg);
        }

        .file-icon {
            font-size: 1.1rem;
            width: 20px;
            text-align: center;
        }

        .file-icon.folder { color: #f59e0b; }
        .file-icon.file { color: #3b82f6; }

        .file-name {
            font-weight: 500;
            white-space: nowrap;
            overflow: hidden;
            text-overflow: ellipsis;
            cursor: pointer;
        }

        .file-name:hover {
            text-decoration: underline;
        }

        .file-size {
            color: var(--text-secondary);
            font-size: 0.8rem;
            min-width: 60px;
            text-align: right;
        }

        .file-actions {
            opacity: 0;
            transition: opacity var(--transition-speed);
            display: flex;
            gap: 0.25rem;
        }

        .file-item:hover .file-actions {
            opacity: 1;
        }

        .action-btn {
            background: transparent;
            border: none;
            color: var(--text-secondary);
            width: 28px;
            height: 28px;
            border-radius: 6px;
            display: inline-flex;
            align-items: center;
            justify-content: center;
            cursor: pointer;
            transition: all var(--transition-speed);
        }

        .action-btn:hover {
            background: rgba(255, 255, 255, 0.1);
            color: var(--text-primary);
        }

        .action-btn.delete:hover {
            background: rgba(239, 68, 68, 0.2);
            color: var(--accent-red);
        }

        .empty-explorer {
            display: flex;
            flex-direction: column;
            align-items: center;
            justify-content: center;
            gap: 0.75rem;
            padding: 4rem 2rem;
            color: var(--text-secondary);
            text-align: center;
            flex-grow: 1;
        }

        .empty-explorer i {
            font-size: 2.5rem;
            opacity: 0.3;
        }

        /* Toast Notifications */
        .toast-container {
            position: fixed;
            top: 1.5rem;
            right: 1.5rem;
            display: flex;
            flex-direction: column;
            gap: 0.75rem;
            z-index: 9999;
            pointer-events: none;
        }

        .toast {
            background: rgba(22, 16, 35, 0.9);
            backdrop-filter: blur(10px);
            border: 1px solid var(--panel-border);
            border-radius: 8px;
            padding: 0.85rem 1.25rem;
            color: var(--text-primary);
            box-shadow: 0 10px 25px rgba(0, 0, 0, 0.4);
            display: flex;
            align-items: center;
            gap: 0.75rem;
            min-width: 280px;
            max-width: 400px;
            transform: translateX(120%);
            transition: transform 0.3s cubic-bezier(0.175, 0.885, 0.32, 1.275);
            pointer-events: auto;
        }

        .toast.show {
            transform: translateX(0);
        }

        .toast-success { border-left: 4px solid var(--accent-green); }
        .toast-error { border-left: 4px solid var(--accent-red); }
        .toast-info { border-left: 4px solid var(--accent-cyan); }

        .toast-icon { font-size: 1.1rem; }
        .toast-success .toast-icon { color: var(--accent-green); }
        .toast-error .toast-icon { color: var(--accent-red); }
        .toast-info .toast-icon { color: var(--accent-cyan); }

        .toast-content {
            font-size: 0.85rem;
            font-weight: 500;
            flex-grow: 1;
        }

        /* Modal Dialog */
        .modal-overlay {
            position: fixed;
            top: 0;
            left: 0;
            right: 0;
            bottom: 0;
            background: rgba(0, 0, 0, 0.6);
            backdrop-filter: blur(4px);
            display: flex;
            align-items: center;
            justify-content: center;
            z-index: 999;
            opacity: 0;
            pointer-events: none;
            transition: opacity var(--transition-speed) ease;
        }

        .modal-overlay.show {
            opacity: 1;
            pointer-events: auto;
        }

        .modal {
            background: rgba(22, 16, 35, 0.95);
            border: 1px solid var(--panel-border);
            border-radius: 12px;
            padding: 1.5rem;
            width: 100%;
            max-width: 400px;
            box-shadow: 0 15px 35px rgba(0, 0, 0, 0.5);
            transform: scale(0.9);
            transition: transform var(--transition-speed) ease;
        }

        .modal-overlay.show .modal {
            transform: scale(1);
        }

        .modal-title {
            font-size: 1.1rem;
            font-weight: 600;
            margin-bottom: 0.75rem;
        }

        .modal-input {
            width: 100%;
            background: rgba(0, 0, 0, 0.2);
            border: 1px solid var(--panel-border);
            border-radius: 6px;
            padding: 0.6rem 0.8rem;
            color: var(--text-primary);
            font-family: inherit;
            font-size: 0.9rem;
            outline: none;
            margin-bottom: 1.25rem;
        }

        .modal-input:focus {
            border-color: var(--accent-purple);
        }

        .modal-buttons {
            display: flex;
            justify-content: flex-end;
            gap: 0.5rem;
        }
    </style>
</head>
<body>
    <header>
        <div class="logo">
            <i class="fa-solid fa-cloud-arrow-up"></i>
            <h1>CNB 工作区上传器</h1>
            <span>v1.0</span>
        </div>
        <div>
            <button class="btn btn-secondary btn-sm" onclick="refreshExplorer()"><i class="fa-solid fa-sync"></i> 刷新工作区</button>
        </div>
    </header>

    <div class="container">
        <!-- Left Side: Upload Panel -->
        <div class="glass-panel">
            <h2 class="panel-title">上传文件与文件夹</h2>
            
            <div class="dropzone" id="dropzone">
                <i class="fa-solid fa-folder-open dropzone-icon"></i>
                <div class="dropzone-text">
                    <h3>将文件或文件夹拖拽到此处</h3>
                    <p>或点击下方按钮进行选择</p>
                </div>
                <div class="upload-options">
                    <button class="btn btn-primary" onclick="triggerFileInput()"><i class="fa-solid fa-file"></i> 选择文件</button>
                    <button class="btn btn-secondary" onclick="triggerFolderInput()"><i class="fa-solid fa-folder-plus"></i> 选择文件夹</button>
                </div>
                <!-- Hidden inputs -->
                <input type="file" id="file-input" multiple style="display: none;" onchange="handleFileSelect(event)">
                <input type="file" id="folder-input" webkitdirectory directory multiple style="display: none;" onchange="handleFileSelect(event)">
            </div>

            <!-- Upload Queue list -->
            <div class="queue-container">
                <div class="panel-title">
                    <span>上传队列</span>
                    <button class="btn btn-secondary btn-sm" id="clear-queue-btn" onclick="clearQueue()" style="display: none;">清空队列</button>
                </div>

                <div class="queue-stats" id="queue-stats" style="display: none;">
                    <div class="stat-item">
                        <span class="stat-label">总进度</span>
                        <span class="stat-val" id="stat-progress">0%</span>
                    </div>
                    <div class="stat-item">
                        <span class="stat-label">已上传</span>
                        <span class="stat-val" id="stat-size">0 / 0 MB</span>
                    </div>
                    <div class="stat-item">
                        <span class="stat-label">平均速度</span>
                        <span class="stat-val" id="stat-speed">0 KB/s</span>
                    </div>
                    <div class="stat-item">
                        <span class="stat-label">剩余时间 (ETA)</span>
                        <span class="stat-val" id="stat-eta">--:--</span>
                    </div>
                </div>

                <div class="progress-bar-container" id="overall-progress-container" style="display: none;">
                    <div class="progress-bar" id="overall-progress-bar"></div>
                </div>

                <ul class="queue-list" id="queue-list">
                    <!-- Dynamic Queue Items -->
                    <div class="empty-explorer" id="queue-empty-state">
                        <i class="fa-solid fa-cloud-arrow-up"></i>
                        <p>暂无上传任务。选择文件或文件夹即可开始上传。</p>
                    </div>
                </ul>
            </div>
        </div>

        <!-- Right Side: Workspace File Explorer -->
        <div class="glass-panel">
            <h2 class="panel-title">工作区浏览器</h2>
            
            <div class="explorer-header">
                <div class="breadcrumbs" id="breadcrumbs">
                    <span class="breadcrumb-item" onclick="navigateTo('')">根目录</span>
                </div>
                <div class="explorer-actions">
                    <button class="btn btn-secondary btn-sm" onclick="showMkdirModal()"><i class="fa-solid fa-folder-plus"></i> 新建文件夹</button>
                </div>
            </div>

            <div class="explorer-search">
                <i class="fa-solid fa-magnifying-glass"></i>
                <input type="text" id="explorer-search-input" placeholder="在当前目录搜索文件..." oninput="filterFiles()">
            </div>

            <div class="file-list-container">
                <div class="file-list-header">
                    <div></div>
                    <div>名称</div>
                    <div>大小</div>
                    <div>操作</div>
                </div>
                <ul class="file-list" id="file-list">
                    <!-- Dynamic Files -->
                </ul>
                <div class="empty-explorer" id="explorer-empty-state" style="display: none;">
                    <i class="fa-solid fa-folder-open"></i>
                    <p>该文件夹为空。</p>
                </div>
            </div>
        </div>
    </div>

    <!-- Modals & Toasts -->
    <div class="toast-container" id="toast-container"></div>

    <div class="modal-overlay" id="mkdir-modal">
        <div class="modal">
            <div class="modal-title">创建新文件夹</div>
            <input type="text" class="modal-input" id="mkdir-input" placeholder="文件夹名称">
            <div class="modal-buttons">
                <button class="btn btn-secondary btn-sm" onclick="hideMkdirModal()">取消</button>
                <button class="btn btn-primary btn-sm" onclick="createFolder()">创建</button>
            </div>
        </div>
    </div>

    <script>
        let currentDirectory = '';
        let allFiles = [];
        let uploadQueue = null;

        // Init
        document.addEventListener('DOMContentLoaded', () => {
            refreshExplorer();
            setupDragAndDrop();
        });

        // Toast Notification System
        function showToast(message, type = 'info') {
            const container = document.getElementById('toast-container');
            const toast = document.createElement('div');
            toast.className = `toast toast-${type}`;
            
            let iconClass = 'fa-circle-info';
            if (type === 'success') iconClass = 'fa-circle-check';
            if (type === 'error') iconClass = 'fa-circle-exclamation';

            toast.innerHTML = `
                <i class="fa-solid ${iconClass} toast-icon"></i>
                <div class="toast-content">${message}</div>
            `;
            
            container.appendChild(toast);
            
            // Trigger animation
            setTimeout(() => toast.classList.add('show'), 10);
            
            // Remove after 3.5 seconds
            setTimeout(() => {
                toast.classList.remove('show');
                setTimeout(() => toast.remove(), 300);
            }, 3500);
        }

        // File Explorer API helpers
        async function fetchFiles(path = '') {
            try {
                const response = await fetch(`/api/files?path=${encodeURIComponent(path)}`);
                if (!response.ok) throw new Error(await response.text());
                const data = await response.json();
                return data;
            } catch (e) {
                showToast(`加载目录失败: ${e.message}`, 'error');
                return null;
            }
        }

        async function refreshExplorer() {
            const data = await fetchFiles(currentDirectory);
            if (data) {
                currentDirectory = data.current_path;
                allFiles = data.items;
                renderExplorer();
                renderBreadcrumbs();
            }
        }

        function renderBreadcrumbs() {
            const breadcrumbs = document.getElementById('breadcrumbs');
            breadcrumbs.innerHTML = `<span class="breadcrumb-item" onclick="navigateTo('')"><i class="fa-solid fa-house"></i> 根目录</span>`;
            
            if (!currentDirectory) return;

            const parts = currentDirectory.split('/');
            let accumulatedPath = '';
            
            parts.forEach((part) => {
                if (!part) return;
                accumulatedPath += (accumulatedPath ? '/' : '') + part;
                const pathForBinding = accumulatedPath; // closure
                breadcrumbs.innerHTML += `
                    <span class="breadcrumb-separator"><i class="fa-solid fa-chevron-right"></i></span>
                    <span class="breadcrumb-item" onclick="navigateTo('${pathForBinding}')">${part}</span>
                `;
            });
        }

        function renderExplorer() {
            const fileList = document.getElementById('file-list');
            const emptyState = document.getElementById('explorer-empty-state');
            const filterText = document.getElementById('explorer-search-input').value.toLowerCase().trim();
            
            const filtered = allFiles.filter(item => item.name.toLowerCase().includes(filterText));
            
            if (filtered.length === 0) {
                fileList.innerHTML = '';
                emptyState.style.display = 'flex';
                return;
            } else {
                emptyState.style.display = 'none';
            }

            let html = '';

            // Render up level if not at root
            if (currentDirectory) {
                const parentDir = currentDirectory.substring(0, currentDirectory.lastIndexOf('/'));
                html += `
                    <li class="file-item">
                        <div class="file-icon"><i class="fa-solid fa-folder-open"></i></div>
                        <div class="file-name" onclick="navigateTo('${parentDir.replace(/'/g, "\\'")}')">.. (返回上级目录)</div>
                        <div class="file-size">-</div>
                        <div class="file-actions"></div>
                    </li>
                `;
            }

            filtered.forEach(item => {
                const icon = item.is_dir 
                    ? '<i class="fa-solid fa-folder file-icon folder"></i>' 
                    : '<i class="fa-solid fa-file file-icon file"></i>';
                
                const sizeStr = item.is_dir ? '-' : formatBytes(item.size);
                
                const escapedPath = item.path.replace(/'/g, "\\'");
                const escapedName = item.name.replace(/'/g, "\\'");
                
                // Clicking logic
                const clickHandler = item.is_dir 
                    ? `navigateTo('${escapedPath}')` 
                    : `showToast('文件: ${escapedName} (${sizeStr})', 'info')`;
                
                html += `
                    <li class="file-item">
                        <div class="file-icon">${icon}</div>
                        <div class="file-name" onclick="${clickHandler}">${item.name}</div>
                        <div class="file-size">${sizeStr}</div>
                        <div class="file-actions">
                            <button class="action-btn delete" onclick="deleteItem('${escapedPath}')" title="删除"><i class="fa-regular fa-trash-can"></i></button>
                        </div>
                    </li>
                `;
            });

            fileList.innerHTML = html;
        }

        function navigateTo(path) {
            currentDirectory = path;
            document.getElementById('explorer-search-input').value = '';
            refreshExplorer();
        }

        function filterFiles() {
            renderExplorer();
        }

        async function deleteItem(path) {
            if (!confirm(`您确定要删除 "${path.split('/').pop()}" 吗？`)) return;
            
            try {
                const response = await fetch(`/api/delete?path=${encodeURIComponent(path)}`, { method: 'DELETE' });
                if (!response.ok) throw new Error(await response.text());
                showToast('项目删除成功', 'success');
                refreshExplorer();
            } catch (e) {
                showToast(`删除失败: ${e.message}`, 'error');
            }
        }

        // Folder Creation
        function showMkdirModal() {
            document.getElementById('mkdir-modal').classList.add('show');
            document.getElementById('mkdir-input').focus();
        }

        function hideMkdirModal() {
            document.getElementById('mkdir-modal').classList.remove('show');
            document.getElementById('mkdir-input').value = '';
        }

        async function createFolder() {
            const folderName = document.getElementById('mkdir-input').value.trim();
            if (!folderName) return;
            
            const fullPath = currentDirectory ? `${currentDirectory}/${folderName}` : folderName;
            
            try {
                const formData = new FormData();
                formData.append('path', fullPath);
                
                const response = await fetch('/api/mkdir', {
                    method: 'POST',
                    body: formData
                });
                
                if (!response.ok) throw new Error(await response.text());
                
                showToast(`成功创建文件夹 "${folderName}"`, 'success');
                hideMkdirModal();
                refreshExplorer();
            } catch (e) {
                showToast(`创建文件夹失败: ${e.message}`, 'error');
            }
        }

        // Upload Queue Logic
        const CHUNK_SIZE = 16 * 1024 * 1024;  // 16 MB per chunk
        const CHUNK_CONCURRENCY = 4;           // parallel chunk requests per file
        const FILE_CONCURRENCY = 6;            // parallel files

        class UploadQueue {
            constructor(concurrency = FILE_CONCURRENCY) {
                this.concurrency = concurrency;
                this.queue = [];
                this.active = 0;
                this.totalBytes = 0;
                this.uploadedBytes = 0;
                this.startTime = null;
            }

            add(file, relativePath) {
                let targetPath = relativePath;
                if (currentDirectory) {
                    targetPath = `${currentDirectory}/${relativePath}`;
                }
                
                const queueItem = {
                    id: Math.random().toString(36).substring(2, 9),
                    file,
                    relativePath: targetPath,
                    status: 'pending',
                    progress: 0,
                    bytesUploaded: 0,
                    xhr: null
                };
                
                this.queue.push(queueItem);
                this.totalBytes += file.size;
                // Note: We do not call renderQueueItem here for every file to avoid freezing the browser on bulk adds (1000+ files)
            }

            start() {
                if (this.startTime === null) {
                    this.startTime = Date.now();
                    document.getElementById('queue-stats').style.display = 'grid';
                    document.getElementById('overall-progress-container').style.display = 'block';
                    document.getElementById('clear-queue-btn').style.display = 'inline-flex';
                    
                    const emptyState = document.getElementById('queue-empty-state');
                    if (emptyState) emptyState.remove();
                }
                this.processNext();
            }

            processNext() {
                const isAllFinished = this.queue.every(item => item.status === 'success' || item.status === 'error');
                if (isAllFinished) {
                    showToast('所有文件上传已完成！', 'success');
                    refreshExplorer();
                    return;
                }

                while (this.active < this.concurrency) {
                    const nextItem = this.queue.find(item => item.status === 'pending');
                    if (!nextItem) break;
                    
                    nextItem.status = 'uploading';
                    this.active++;
                    this.renderQueueItem(nextItem); // Only render once it starts uploading
                    this.updateQueueItemUI(nextItem);
                    this.uploadFile(nextItem);
                }
            }

            async uploadFile(item) {
                const totalChunks = Math.max(1, Math.ceil(item.file.size / CHUNK_SIZE));
                
                // Generate a stable uploadId based on file properties
                let hashStr = `${item.file.name}-${item.file.size}-${item.file.lastModified}`;
                let hash = 0;
                for (let i = 0; i < hashStr.length; i++) {
                    hash = (hash << 5) - hash + hashStr.charCodeAt(i);
                    hash |= 0;
                }
                const uploadId = 'up_' + Math.abs(hash).toString(36);
                
                let completedChunks = [];
                try {
                    const checkUrl = `/api/upload/check?upload_id=${uploadId}&relative_path=${encodeURIComponent(item.relativePath)}`;
                    const checkRes = await fetch(checkUrl);
                    if (checkRes.ok) {
                        const checkData = await checkRes.json();
                        if (checkData.exists && checkData.size === item.file.size) {
                            // File exists and size matches, skip upload
                            item.status = 'success';
                            item.progress = 100;
                            this.updateQueueItemUI(item);
                            setTimeout(() => this.fadeOutAndRemove(item.id), 2000);
                            this.active--;
                            this.updateOverallStats();
                            this.processNext();
                            return;
                        }
                        completedChunks = checkData.completed_chunks || [];
                    }
                } catch (e) {
                    console.warn('Check upload status failed:', e);
                }

                // If all chunks exist on disk but file was not merged, re-upload the last chunk to trigger merge
                if (completedChunks.length === totalChunks && totalChunks > 0) {
                    completedChunks = completedChunks.filter(idx => idx !== totalChunks - 1);
                }

                // Per-chunk bytes-uploaded tracker for progress accounting.
                const chunkSent = new Array(totalChunks).fill(0);
                
                // Account for already completed chunks
                completedChunks.forEach(idx => {
                    if (idx < totalChunks) {
                        const start = idx * CHUNK_SIZE;
                        const end = Math.min(start + CHUNK_SIZE, item.file.size);
                        const size = end - start;
                        chunkSent[idx] = size;
                        item.bytesUploaded += size;
                        this.uploadedBytes += size;
                    }
                });

                try {
                    // Send chunks in windows of CHUNK_CONCURRENCY
                    for (let base = 0; base < totalChunks; base += CHUNK_CONCURRENCY) {
                        const window = [];
                        for (let i = base; i < Math.min(base + CHUNK_CONCURRENCY, totalChunks); i++) {
                            if (!completedChunks.includes(i)) {
                                window.push(i);
                            }
                        }
                        await Promise.all(window.map(idx => {
                            const start = idx * CHUNK_SIZE;
                            const end = Math.min(start + CHUNK_SIZE, item.file.size);
                            const chunk = item.file.slice(start, end);
                            return this.uploadChunk(item, chunk, idx, totalChunks, uploadId, chunkSent);
                        }));
                    }
                    item.status = 'success';
                    item.progress = 100;
                    this.updateQueueItemUI(item);
                    setTimeout(() => this.fadeOutAndRemove(item.id), 2000);
                } catch (err) {
                    item.status = 'error';
                    item.progress = 0;
                    item.errorMsg = err.message || '上传失败';
                    this.updateQueueItemUI(item);
                } finally {
                    this.active--;
                    this.updateOverallStats();
                    this.processNext();
                }
            }

            uploadChunk(item, chunk, chunkIndex, totalChunks, uploadId, chunkSent) {
                return new Promise((resolve, reject) => {
                    const formData = new FormData();
                    formData.append('file', chunk);
                    formData.append('relative_path', item.relativePath);
                    formData.append('chunk_index', chunkIndex);
                    formData.append('total_chunks', totalChunks);
                    formData.append('upload_id', uploadId);

                    const xhr = new XMLHttpRequest();
                    // Store all in-flight XHRs so cancelAll() can abort them.
                    if (!item.xhrs) item.xhrs = new Set();
                    item.xhrs.add(xhr);
                    xhr.open('POST', '/api/upload', true);

                    xhr.upload.onprogress = (e) => {
                        if (e.lengthComputable) {
                            const prev = chunkSent[chunkIndex];
                            const diff = e.loaded - prev;
                            chunkSent[chunkIndex] = e.loaded;
                            item.bytesUploaded += diff;
                            this.uploadedBytes += diff;
                            item.progress = Math.min(100, Math.round((item.bytesUploaded / item.file.size) * 100));
                            this.updateQueueItemUI(item);
                            this.updateOverallStats();
                        }
                    };

                    xhr.onload = () => {
                        item.xhrs && item.xhrs.delete(xhr);
                        if (xhr.status === 200) {
                            resolve();
                        } else {
                            try {
                                const err = JSON.parse(xhr.responseText);
                                reject(new Error(err.detail || `HTTP ${xhr.status}`));
                            } catch {
                                reject(new Error(`HTTP ${xhr.status}`));
                            }
                        }
                    };

                    xhr.onerror = () => {
                        item.xhrs && item.xhrs.delete(xhr);
                        reject(new Error('网络错误'));
                    };

                    xhr.send(formData);
                });
            }

            renderQueueItem(item) {
                const queueList = document.getElementById('queue-list');
                const emptyState = document.getElementById('queue-empty-state');
                if (emptyState) emptyState.remove();

                const li = document.createElement('li');
                li.className = 'queue-item';
                li.id = `queue-item-${item.id}`;
                li.innerHTML = `
                    <div class="queue-item-header">
                        <span class="queue-item-name" title="${item.relativePath}">${item.relativePath}</span>
                        <span class="queue-item-status status-pending" id="status-badge-${item.id}">等待中</span>
                    </div>
                    <div class="queue-item-meta">
                        <span id="progress-text-${item.id}">0%</span>
                        <span>&bull;</span>
                        <span>${formatBytes(item.file.size)}</span>
                        <span id="error-text-${item.id}" style="color: var(--accent-red); margin-left: auto; display: none;"></span>
                    </div>
                    <div class="queue-item-progress-bg">
                        <div class="queue-item-progress" id="progress-bar-${item.id}"></div>
                    </div>
                `;
                queueList.appendChild(li);
                queueList.scrollTop = queueList.scrollHeight;
            }

            fadeOutAndRemove(itemId) {
                const el = document.getElementById(`queue-item-${itemId}`);
                if (el) {
                    el.style.transition = 'opacity 0.5s ease, height 0.5s ease, margin 0.5s ease, padding 0.5s ease';
                    el.style.opacity = '0';
                    el.style.height = '0';
                    el.style.padding = '0';
                    el.style.margin = '0';
                    el.style.border = 'none';
                    setTimeout(() => {
                        el.remove();
                        const queueList = document.getElementById('queue-list');
                        if (queueList && queueList.children.length === 0) {
                            const isAllFinished = this.queue.every(item => item.status === 'success' || item.status === 'error');
                            if (isAllFinished) {
                                queueList.innerHTML = `
                                    <div class="empty-explorer" id="queue-empty-state">
                                        <i class="fa-solid fa-cloud-arrow-up"></i>
                                        <p>所有任务已完成！</p>
                                    </div>
                                `;
                            }
                        }
                    }, 500);
                }
            }

            updateQueueItemUI(item) {
                const badge = document.getElementById(`status-badge-${item.id}`);
                const progressBar = document.getElementById(`progress-bar-${item.id}`);
                const progressText = document.getElementById(`progress-text-${item.id}`);
                const errorText = document.getElementById(`error-text-${item.id}`);

                const statusMap = {
                    'pending': '等待中',
                    'uploading': '上传中',
                    'success': '成功',
                    'error': '失败'
                };

                if (badge) {
                    badge.className = `queue-item-status status-${item.status}`;
                    badge.textContent = statusMap[item.status] || item.status.toUpperCase();
                }

                if (progressBar) {
                    progressBar.style.width = `${item.progress}%`;
                    if (item.status === 'success') {
                        progressBar.style.backgroundColor = 'var(--accent-green)';
                    } else if (item.status === 'error') {
                        progressBar.style.backgroundColor = 'var(--accent-red)';
                    }
                }

                if (progressText) {
                    progressText.textContent = `${item.progress}%`;
                }

                if (item.status === 'error' && errorText) {
                    errorText.textContent = item.errorMsg;
                    errorText.style.display = 'inline';
                }
            }

            updateOverallStats() {
                const pct = this.totalBytes > 0 ? Math.round((this.uploadedBytes / this.totalBytes) * 100) : 0;
                document.getElementById('stat-progress').textContent = `${pct}%`;
                document.getElementById('overall-progress-bar').style.width = `${pct}%`;
                
                const mbUploaded = (this.uploadedBytes / (1024 * 1024)).toFixed(1);
                const mbTotal = (this.totalBytes / (1024 * 1024)).toFixed(1);
                const completedCount = this.queue.filter(item => item.status === 'success' || item.status === 'error').length;
                const totalCount = this.queue.length;
                document.getElementById('stat-size').textContent = `${mbUploaded} / ${mbTotal} MB (${completedCount} / ${totalCount} 文件)`;

                // Calculate Speed and ETA
                const elapsed = (Date.now() - this.startTime) / 1000; // secs
                if (elapsed > 0.5) {
                    const speedBytes = this.uploadedBytes / elapsed; // bytes/sec
                    document.getElementById('stat-speed').textContent = `${formatSpeed(speedBytes)}`;

                    const remainingBytes = this.totalBytes - this.uploadedBytes;
                    const etaSecs = speedBytes > 0 ? remainingBytes / speedBytes : 0;
                    document.getElementById('stat-eta').textContent = formatTime(etaSecs);
                }
            }

            cancelAll() {
                this.queue.forEach(item => {
                    if (item.xhrs) {
                        item.xhrs.forEach(xhr => xhr.abort());
                        item.xhrs.clear();
                    }
                });
            }
        }

        function triggerFileInput() {
            document.getElementById('file-input').click();
        }

        function triggerFolderInput() {
            document.getElementById('folder-input').click();
        }

        function handleFileSelect(event) {
            const files = event.target.files;
            if (files.length === 0) return;

            if (!uploadQueue) {
                uploadQueue = new UploadQueue();
            }

            for (let i = 0; i < files.length; i++) {
                const file = files[i];
                // Resolve relative path for folders
                let relPath = file.webkitRelativePath || file.name;
                uploadQueue.add(file, relPath);
            }

            // Clear inputs so same selection triggers event next time
            event.target.value = '';

            uploadQueue.start();
        }

        function clearQueue() {
            if (uploadQueue) {
                uploadQueue.cancelAll();
            }
            
            document.getElementById('queue-list').innerHTML = `
                <div class="empty-explorer" id="queue-empty-state">
                    <i class="fa-solid fa-cloud-arrow-up"></i>
                    <p>暂无上传任务。选择文件或文件夹即可开始上传。</p>
                </div>
            `;
            document.getElementById('queue-stats').style.display = 'none';
            document.getElementById('overall-progress-container').style.display = 'none';
            document.getElementById('clear-queue-btn').style.display = 'none';
            
            uploadQueue = null;
        }

        // Drag and Drop
        function setupDragAndDrop() {
            const dropzone = document.getElementById('dropzone');

            ['dragenter', 'dragover'].forEach(eventName => {
                dropzone.addEventListener(eventName, (e) => {
                    e.preventDefault();
                    e.stopPropagation();
                    dropzone.classList.add('dragover');
                }, false);
            });

            ['dragleave', 'drop'].forEach(eventName => {
                dropzone.addEventListener(eventName, (e) => {
                    e.preventDefault();
                    e.stopPropagation();
                    dropzone.classList.remove('dragover');
                }, false);
            });

            dropzone.addEventListener('drop', async (e) => {
                const dt = e.dataTransfer;
                
                // Check if files or items exist
                if (dt.items) {
                    if (!uploadQueue) {
                        uploadQueue = new UploadQueue();
                    }
                    
                    // Webkit Entry API support for drag-and-dropping directories recursively
                    const entries = [];
                    for (let i = 0; i < dt.items.length; i++) {
                        const item = dt.items[i];
                        if (item.kind === 'file') {
                            const entry = item.webkitGetAsEntry();
                            if (entry) {
                                entries.push(entry);
                            }
                        }
                    }
                    
                    if (entries.length > 0) {
                        for (const entry of entries) {
                            await traverseFileTree(entry, '');
                        }
                        uploadQueue.start();
                    }
                } else {
                    // Fallback to standard files if entries are not supported
                    const files = dt.files;
                    if (files.length > 0) {
                        if (!uploadQueue) {
                            uploadQueue = new UploadQueue();
                        }
                        for (let i = 0; i < files.length; i++) {
                            uploadQueue.add(files[i], files[i].name);
                        }
                        uploadQueue.start();
                    }
                }
            }, false);
        }

        // Recursive directory traversal for Drag & Drop
        async function traverseFileTree(entry, path) {
            if (entry.isFile) {
                const file = await getFileFromEntry(entry);
                const relPath = path ? `${path}/${file.name}` : file.name;
                uploadQueue.add(file, relPath);
            } else if (entry.isDirectory) {
                const dirReader = entry.createReader();
                const entries = await readAllEntries(dirReader);
                const nextPath = path ? `${path}/${entry.name}` : entry.name;
                for (const childEntry of entries) {
                    await traverseFileTree(childEntry, nextPath);
                }
            }
        }

        function getFileFromEntry(entry) {
            return new Promise((resolve, reject) => {
                entry.file(resolve, reject);
            });
        }

        function readAllEntries(dirReader) {
            return new Promise((resolve, reject) => {
                let allEntries = [];
                function read() {
                    dirReader.readEntries((entries) => {
                        if (entries.length === 0) {
                            resolve(allEntries);
                        } else {
                            allEntries = allEntries.concat(entries);
                            read();
                        }
                    }, reject);
                }
                read();
            });
        }

        // Formats
        function formatBytes(bytes) {
            if (bytes === 0) return '0 B';
            const k = 1024;
            const sizes = ['B', 'KB', 'MB', 'GB', 'TB'];
            const i = Math.floor(Math.log(bytes) / Math.log(k));
            return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
        }

        function formatSpeed(bytesPerSec) {
            if (bytesPerSec === 0) return '0 B/s';
            const k = 1024;
            const sizes = ['B/s', 'KB/s', 'MB/s', 'GB/s'];
            const i = Math.floor(Math.log(bytesPerSec) / Math.log(k));
            return parseFloat((bytesPerSec / Math.pow(k, i)).toFixed(1)) + ' ' + sizes[i];
        }

        // formatTime formats seconds into readable time
        function formatTime(seconds) {
            if (!isFinite(seconds) || seconds <= 0) return '--:--';
            const m = Math.floor(seconds / 60);
            const s = Math.floor(seconds % 60);
            return `${m}:${s.toString().padStart(2, '0')}`;
        }
    </script>
</body>
</html>"""
