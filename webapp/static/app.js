// CpGPT Web Application - Frontend JavaScript

let selectedFile = null;
let currentTaskId = null;
let pollInterval = null;

// 文件选择处理
document.getElementById('fileInput').addEventListener('change', function(e) {
    const file = e.target.files[0];
    if (file) {
        selectedFile = file;
        showFileInfo(file);
    }
});

// 拖拽上传
const uploadSection = document.getElementById('uploadSection');

uploadSection.addEventListener('dragover', function(e) {
    e.preventDefault();
    uploadSection.classList.add('dragover');
});

uploadSection.addEventListener('dragleave', function(e) {
    e.preventDefault();
    uploadSection.classList.remove('dragover');
});

uploadSection.addEventListener('drop', function(e) {
    e.preventDefault();
    uploadSection.classList.remove('dragover');
    
    const file = e.dataTransfer.files[0];
    if (file) {
        // 验证文件格式
        const validExtensions = ['.csv', '.arrow', '.feather'];
        const fileExtension = '.' + file.name.split('.').pop().toLowerCase();
        
        if (validExtensions.includes(fileExtension)) {
            selectedFile = file;
            document.getElementById('fileInput').files = e.dataTransfer.files;
            showFileInfo(file);
        } else {
            showError('不支持的文件格式。请上传CSV或Arrow格式的文件。');
        }
    }
});

// 显示文件信息
function showFileInfo(file) {
    document.getElementById('fileName').textContent = file.name;
    document.getElementById('fileSize').textContent = formatFileSize(file.size);
    document.getElementById('fileInfo').classList.add('show');
}

// 取消文件选择
function cancelFile() {
    selectedFile = null;
    document.getElementById('fileInput').value = '';
    document.getElementById('fileInfo').classList.remove('show');
}

// 格式化文件大小
function formatFileSize(bytes) {
    if (bytes === 0) return '0 Bytes';
    const k = 1024;
    const sizes = ['Bytes', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return Math.round(bytes / Math.pow(k, i) * 100) / 100 + ' ' + sizes[i];
}

// 上传文件
async function uploadFile() {
    if (!selectedFile) {
        showError('请先选择文件');
        return;
    }

    // 隐藏上传区域，显示进度
    document.getElementById('uploadSection').style.display = 'none';
    document.getElementById('progressSection').classList.add('show');
    hideError();

    const formData = new FormData();
    formData.append('file', selectedFile);

    try {
        const response = await fetch('/api/upload', {
            method: 'POST',
            body: formData
        });

        const data = await response.json();

        if (response.ok && data.success) {
            currentTaskId = data.task_id;
            // 开始轮询任务状态
            startPolling();
        } else {
            throw new Error(data.detail || '文件上传失败');
        }
    } catch (error) {
        showError('上传失败: ' + error.message);
        resetUploadSection();
    }
}

// 开始轮询任务状态
function startPolling() {
    pollInterval = setInterval(checkTaskStatus, 2000); // 每2秒检查一次
}

// 停止轮询
function stopPolling() {
    if (pollInterval) {
        clearInterval(pollInterval);
        pollInterval = null;
    }
}

// 检查任务状态
async function checkTaskStatus() {
    if (!currentTaskId) return;

    try {
        const response = await fetch(`/api/task/${currentTaskId}`);
        const task = await response.json();

        if (response.ok) {
            updateProgress(task);

            if (task.status === 'completed') {
                stopPolling();
                showResult();
            } else if (task.status === 'failed') {
                stopPolling();
                showError('分析失败: ' + (task.error || '未知错误'));
                resetUploadSection();
            }
        } else {
            throw new Error('无法获取任务状态');
        }
    } catch (error) {
        console.error('状态检查错误:', error);
    }
}

// 更新进度显示
function updateProgress(task) {
    const progressBar = document.getElementById('progressBar');
    const statusText = document.getElementById('statusText');

    progressBar.style.width = task.progress + '%';
    progressBar.textContent = task.progress + '%';
    statusText.textContent = task.message;

    // 根据进度更新图标
    const statusIcon = document.querySelector('.progress-message .status-icon');
    if (task.progress < 30) {
        statusIcon.textContent = '⏳';
    } else if (task.progress < 60) {
        statusIcon.textContent = '🔬';
    } else if (task.progress < 90) {
        statusIcon.textContent = '📊';
    } else {
        statusIcon.textContent = '✨';
    }
}

// 显示结果
function showResult() {
    document.getElementById('progressSection').classList.remove('show');
    document.getElementById('resultSection').classList.add('show');
}

// 查看报告
function viewReport() {
    if (!currentTaskId) return;

    const reportUrl = `/results/${currentTaskId}/analysis_report.html`;
    const reportFrame = document.getElementById('reportFrame');
    const reportPreview = document.getElementById('reportPreview');

    reportFrame.src = reportUrl;
    reportPreview.classList.add('show');

    // 滚动到报告位置
    reportPreview.scrollIntoView({ behavior: 'smooth' });
}

// 下载PDF
async function downloadPDF() {
    if (!currentTaskId) return;

    try {
        // 显示加载提示
        const btn = event.target;
        const originalText = btn.textContent;
        btn.textContent = '生成中...';
        btn.disabled = true;

        const response = await fetch(`/api/download/${currentTaskId}/pdf`);

        if (response.ok) {
            const blob = await response.blob();
            const url = window.URL.createObjectURL(blob);
            const a = document.createElement('a');
            a.href = url;
            a.download = `cpgpt_analysis_${currentTaskId}.pdf`;
            document.body.appendChild(a);
            a.click();
            window.URL.revokeObjectURL(url);
            document.body.removeChild(a);
        } else {
            const error = await response.json();
            throw new Error(error.detail || 'PDF下载失败');
        }

        btn.textContent = originalText;
        btn.disabled = false;
    } catch (error) {
        showError('PDF下载失败: ' + error.message);
        event.target.textContent = '下载PDF';
        event.target.disabled = false;
    }
}

// 重置页面
function resetPage() {
    // 重置所有状态
    selectedFile = null;
    currentTaskId = null;
    stopPolling();

    // 重置UI
    document.getElementById('fileInput').value = '';
    document.getElementById('fileInfo').classList.remove('show');
    document.getElementById('progressSection').classList.remove('show');
    document.getElementById('resultSection').classList.remove('show');
    document.getElementById('reportPreview').classList.remove('show');
    document.getElementById('uploadSection').style.display = 'block';
    
    // 重置进度条
    document.getElementById('progressBar').style.width = '0%';
    document.getElementById('progressBar').textContent = '0%';
    
    hideError();
}

// 重置上传区域
function resetUploadSection() {
    document.getElementById('uploadSection').style.display = 'block';
    document.getElementById('progressSection').classList.remove('show');
}

// 显示错误
function showError(message) {
    const errorDiv = document.getElementById('errorMessage');
    errorDiv.textContent = '❌ ' + message;
    errorDiv.classList.add('show');
}

// 隐藏错误
function hideError() {
    const errorDiv = document.getElementById('errorMessage');
    errorDiv.classList.remove('show');
}

// 页面加载完成后的初始化
document.addEventListener('DOMContentLoaded', function() {
    console.log('CpGPT Web Application loaded');
    
    // 检查后端健康状态
    fetch('/health')
        .then(response => response.json())
        .then(data => {
            console.log('Backend health:', data);
            if (!data.gpu_available) {
                console.warn('GPU not available - analysis may be slower');
            }
        })
        .catch(error => {
            console.error('Backend health check failed:', error);
        });
});

