import sys
import os
import json
import cv2
import numpy as np
from datetime import datetime
import warnings
from typing import Dict, List, Tuple, Optional
warnings.filterwarnings("ignore")

from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, 
    QLabel, QPushButton, QFrame, QScrollArea, QTreeWidget, 
    QTreeWidgetItem, QFileDialog, QMessageBox, QDialog, QSlider, 
    QLineEdit, QSplitter, QSpinBox, QGraphicsView, QGraphicsScene,
    QAbstractItemView, QFormLayout, QGroupBox, QCheckBox, QComboBox
)
from PySide6.QtCore import Qt, QTimer, Signal, QPoint, QRectF, QThread
from PySide6.QtGui import QPixmap, QPainter, QPen, QBrush, QColor, QFont

class BubbleDetector:
    """개선된 버블 감지 클래스"""
    
    def __init__(self):
        self.default_params = {
            'min_dist': 20,
            'min_radius': 5,
            'max_radius': 25,
            'sensitivity': 60,
            'blur_kernel': 5
        }
    
    def preprocess_image(self, image: np.ndarray) -> np.ndarray:
        """이미지 전처리"""
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        
        # 적응적 히스토그램 평활화로 조명 보정
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)
        
        # 가우시안 블러로 노이즈 제거
        blurred = cv2.GaussianBlur(enhanced, (5, 5), 0)
        
        return blurred
    
    def detect_bubbles(self, image: np.ndarray, params: Dict = None) -> List[Tuple[int, int, int]]:
        """버블 감지 (개선된 알고리즘)"""
        if params is None:
            params = self.default_params.copy()
        
        processed = self.preprocess_image(image)
        
        # HoughCircles 파라미터 계산
        sensitivity = params.get('sensitivity', 60)
        param1 = 50
        param2 = max(10, 40 - (sensitivity - 50) * 0.5)
        
        circles = cv2.HoughCircles(
            processed,
            cv2.HOUGH_GRADIENT,
            dp=1,
            minDist=params.get('min_dist', 20),
            param1=param1,
            param2=int(param2),
            minRadius=params.get('min_radius', 5),
            maxRadius=params.get('max_radius', 25)
        )
        
        if circles is not None:
            circles = np.round(circles[0, :]).astype("int")
            return [(x, y, r) for x, y, r in circles]
        
        return []
    
    def validate_bubble_region(self, image: np.ndarray, x: int, y: int, radius: int) -> bool:
        """버블 영역 검증 (형태학적 분석)"""
        try:
            # ROI 추출
            x1, y1 = max(0, x - radius), max(0, y - radius)
            x2, y2 = min(image.shape[1], x + radius), min(image.shape[0], y + radius)
            roi = image[y1:y2, x1:x2]
            
            if roi.size == 0:
                return False
            
            # 이진화
            gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY) if len(roi.shape) == 3 else roi
            _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
            
            # 원형도 검사
            contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if not contours:
                return False
            
            largest_contour = max(contours, key=cv2.contourArea)
            area = cv2.contourArea(largest_contour)
            perimeter = cv2.arcLength(largest_contour, True)
            
            if perimeter == 0:
                return False
            
            circularity = 4 * np.pi * area / (perimeter * perimeter)
            return 0.5 < circularity < 1.3  # 원형도 기준
            
        except Exception:
            return False

class ImageRotationCorrector:
    """이미지 회전 보정 클래스"""
    
    @staticmethod
    def detect_skew_angle(image: np.ndarray) -> float:
        """기울기 각도 감지"""
        try:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
            
            # 엣지 검출
            edges = cv2.Canny(gray, 50, 150, apertureSize=3)
            
            # 허프 변환으로 선 검출
            lines = cv2.HoughLines(edges, 1, np.pi/180, threshold=100)
            
            if lines is None:
                return 0.0
            
            # 각도 계산
            angles = []
            for rho, theta in lines[:10]:  # 상위 10개 선만 사용
                angle = theta * 180 / np.pi
                if angle < 45:
                    angles.append(angle)
                elif angle > 135:
                    angles.append(angle - 180)
            
            if angles:
                return np.median(angles)
            
            return 0.0
            
        except Exception:
            return 0.0
    
    @staticmethod
    def correct_rotation(image: np.ndarray, angle: float) -> np.ndarray:
        """이미지 회전 보정"""
        if abs(angle) < 0.1:  # 0.1도 미만은 보정하지 않음
            return image
        
        height, width = image.shape[:2]
        center = (width // 2, height // 2)
        
        # 회전 행렬 생성
        rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
        
        # 회전 적용
        corrected = cv2.warpAffine(image, rotation_matrix, (width, height), 
                                 flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REFLECT)
        
        return corrected

class SimpleImageCanvas(QGraphicsView):
    """단순화된 이미지 캔버스 (실시간 미리보기 지원)"""
    
    area_selected = Signal(int, int, int, int)  # x1, y1, x2, y2
    
    def __init__(self):
        super().__init__()
        self.scene = QGraphicsScene()
        self.setScene(self.scene)
        self.setRenderHint(QPainter.Antialiasing)
        
        # 상태 변수
        self.original_pixmap = None
        self.selection_start = None
        self.selection_rect = None
        self.is_selecting = False
        self.scale_factor = 1.0
        
        # 마커 관리
        self.permanent_markers = []  # 확정된 마커들
        self.preview_markers = []    # 실시간 미리보기 마커들
        
        self.setMinimumSize(800, 600)
        self.setStyleSheet("background-color: #f0f0f0; border: 1px solid #ccc;")
    
    def set_image(self, image_path: str):
        """이미지 설정"""
        self.original_pixmap = QPixmap(image_path)
        if self.original_pixmap.isNull():
            return False
        
        self.fit_image_to_view()
        return True
    
    def fit_image_to_view(self):
        """이미지를 뷰에 맞춤"""
        if not self.original_pixmap:
            return
        
        view_size = self.viewport().size()
        image_size = self.original_pixmap.size()
        
        # 여백을 고려한 스케일 계산
        margin = 50
        scale_x = (view_size.width() - margin) / image_size.width()
        scale_y = (view_size.height() - margin) / image_size.height()
        self.scale_factor = min(scale_x, scale_y, 1.0)  # 확대는 하지 않음
        
        scaled_size = image_size * self.scale_factor
        scaled_pixmap = self.original_pixmap.scaled(
            scaled_size, Qt.KeepAspectRatio, Qt.SmoothTransformation
        )
        
        self.scene.clear()
        self.scene.addPixmap(scaled_pixmap)
        self.scene.setSceneRect(scaled_pixmap.rect())
        self.centerOn(scaled_pixmap.rect().center())
        
        # 기존 마커들 다시 그리기
        self.redraw_all_markers()
    
    def get_original_coordinates(self, scene_point: QPoint) -> QPoint:
        """씬 좌표를 원본 이미지 좌표로 변환"""
        if self.scale_factor <= 0:
            return scene_point
        
        original_x = int(scene_point.x() / self.scale_factor)
        original_y = int(scene_point.y() / self.scale_factor)
        return QPoint(original_x, original_y)
    
    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton and self.original_pixmap:
            self.selection_start = self.mapToScene(event.pos()).toPoint()
            self.is_selecting = True
            
            if self.selection_rect:
                self.scene.removeItem(self.selection_rect)
                self.selection_rect = None
        
        super().mousePressEvent(event)
    
    def mouseMoveEvent(self, event):
        if self.is_selecting and self.selection_start:
            current_pos = self.mapToScene(event.pos()).toPoint()
            
            if self.selection_rect:
                self.scene.removeItem(self.selection_rect)
            
            rect = QRectF(self.selection_start, current_pos).normalized()
            pen = QPen(QColor("#2196F3"), 2, Qt.DashLine)
            self.selection_rect = self.scene.addRect(rect, pen)
        
        super().mouseMoveEvent(event)
    
    def mouseReleaseEvent(self, event):
        if self.is_selecting and event.button() == Qt.LeftButton:
            end_pos = self.mapToScene(event.pos()).toPoint()
            self.is_selecting = False
            
            if self.selection_rect:
                self.scene.removeItem(self.selection_rect)
                self.selection_rect = None
            
            if self.selection_start:
                # 원본 좌표로 변환
                start_orig = self.get_original_coordinates(self.selection_start)
                end_orig = self.get_original_coordinates(end_pos)
                
                x1 = min(start_orig.x(), end_orig.x())
                y1 = min(start_orig.y(), end_orig.y())
                x2 = max(start_orig.x(), end_orig.x())
                y2 = max(start_orig.y(), end_orig.y())
                
                # 최소 크기 검사
                if (x2 - x1) > 20 and (y2 - y1) > 20:
                    self.area_selected.emit(x1, y1, x2, y2)
                
                self.selection_start = None
        
        super().mouseReleaseEvent(event)
    
    def add_markers(self, positions: List[Tuple[int, int]], color: str = "#4CAF50"):
        """확정 마커 추가 (영구 저장)"""
        new_markers = []
        for x, y in positions:
            scene_x = x * self.scale_factor
            scene_y = y * self.scale_factor
            
            marker = self.scene.addEllipse(scene_x - 6, scene_y - 6, 12, 12,
                                         QPen(QColor(color), 2), QBrush(QColor(color)))
            marker.setData(0, "permanent_marker")
            new_markers.append(marker)
        
        self.permanent_markers.extend(new_markers)
    
    def add_preview_markers(self, positions: List[Tuple[int, int]], color: str = "#FF5722"):
        """🚀 실시간 미리보기 마커 추가 (임시)"""
        for x, y in positions:
            scene_x = x * self.scale_factor
            scene_y = y * self.scale_factor
            
            # 미리보기 마커는 다른 스타일로 표시 (점선 테두리)
            pen = QPen(QColor(color), 3, Qt.DashLine)
            brush = QBrush(QColor(color + "80"))  # 반투명
            
            marker = self.scene.addEllipse(scene_x - 8, scene_y - 8, 16, 16, pen, brush)
            marker.setData(0, "preview_marker")
            self.preview_markers.append(marker)
    
    def clear_preview_markers(self):
        """🧹 실시간 미리보기 마커만 제거"""
        for marker in self.preview_markers:
            if marker.scene():  # 씬에 아직 있는지 확인
                self.scene.removeItem(marker)
        self.preview_markers.clear()
    
    def clear_markers(self):
        """모든 마커 제거"""
        self.clear_preview_markers()
        
        for marker in self.permanent_markers:
            if marker.scene():
                self.scene.removeItem(marker)
        self.permanent_markers.clear()
    
    def redraw_all_markers(self):
        """모든 마커 다시 그리기 (이미지 크기 변경 시)"""
        # 현재 위치 정보를 저장
        permanent_positions = []
        preview_positions = []
        
        # 기존 마커들에서 원본 좌표 추출 (스케일 팩터 역산)
        if self.scale_factor > 0:
            for marker in self.permanent_markers:
                rect = marker.rect()
                center_x = int((rect.center().x()) / self.scale_factor)
                center_y = int((rect.center().y()) / self.scale_factor)
                permanent_positions.append((center_x, center_y))
            
            for marker in self.preview_markers:
                rect = marker.rect()
                center_x = int((rect.center().x()) / self.scale_factor)
                center_y = int((rect.center().y()) / self.scale_factor)
                preview_positions.append((center_x, center_y))
        
        # 기존 마커들 제거
        self.clear_markers()
        
        # 새로운 스케일로 다시 그리기
        if permanent_positions:
            self.add_markers(permanent_positions)
        if preview_positions:
            self.add_preview_markers(preview_positions)
    
    def wheelEvent(self, event):
        """줌 기능"""
        if event.modifiers() == Qt.ControlModifier:
            zoom_factor = 1.15 if event.angleDelta().y() > 0 else 1 / 1.15
            self.scale(zoom_factor, zoom_factor)
        else:
            super().wheelEvent(event)

class SimpleBubbleDialog(QDialog):
    """단순화된 버블 감지 다이얼로그 (실시간 미리보기 포함)"""
    
    def __init__(self, detector: BubbleDetector, image: np.ndarray, roi: Tuple[int, int, int, int], canvas, parent=None):
        super().__init__(parent)
        self.detector = detector
        self.image = image
        self.roi = roi  # x1, y1, x2, y2
        self.canvas = canvas  # 실시간 표시를 위한 캔버스 참조
        self.detected_bubbles = []
        
        self.init_ui()
        self.detect_bubbles()
    
    def init_ui(self):
        self.setWindowTitle("🔍 실시간 버블 감지 및 조정")
        self.setFixedSize(400, 250)
        self.setModal(True)
        
        layout = QVBoxLayout()
        
        # 안내 메시지
        guide_label = QLabel("💡 슬라이더를 움직여서 감지 결과를 실시간으로 확인하세요")
        guide_label.setAlignment(Qt.AlignCenter)
        guide_label.setStyleSheet("color: #666; font-size: 11px; padding: 8px; background-color: #f5f5f5; border-radius: 4px;")
        guide_label.setWordWrap(True)
        layout.addWidget(guide_label)
        
        # 결과 표시
        self.result_label = QLabel("감지 중...")
        self.result_label.setAlignment(Qt.AlignCenter)
        self.result_label.setStyleSheet("font-size: 18px; font-weight: bold; padding: 15px;")
        layout.addWidget(self.result_label)
        
        # 민감도 조정
        sensitivity_group = QFrame()
        sensitivity_group.setStyleSheet("background-color: #f8f9fa; border: 1px solid #dee2e6; border-radius: 6px; padding: 10px;")
        sensitivity_layout = QVBoxLayout()
        
        # 민감도 라벨
        sens_title = QLabel("🎛️ 감지 민감도")
        sens_title.setStyleSheet("font-weight: bold; font-size: 12px; margin-bottom: 5px;")
        sensitivity_layout.addWidget(sens_title)
        
        # 슬라이더 영역
        slider_layout = QHBoxLayout()
        slider_layout.addWidget(QLabel("낮음"))
        
        self.sensitivity_slider = QSlider(Qt.Horizontal)
        self.sensitivity_slider.setRange(20, 100)
        self.sensitivity_slider.setValue(60)
        self.sensitivity_slider.valueChanged.connect(self.on_sensitivity_changed)
        self.sensitivity_slider.setStyleSheet("""
            QSlider::groove:horizontal {
                border: 1px solid #bbb;
                background: white;
                height: 10px;
                border-radius: 4px;
            }
            QSlider::handle:horizontal {
                background: #2196F3;
                border: 1px solid #5c85d6;
                width: 18px;
                margin: -5px 0;
                border-radius: 9px;
            }
        """)
        slider_layout.addWidget(self.sensitivity_slider)
        
        slider_layout.addWidget(QLabel("높음"))
        
        self.sensitivity_label = QLabel("60")
        self.sensitivity_label.setMinimumWidth(30)
        self.sensitivity_label.setStyleSheet("font-weight: bold; color: #2196F3; font-size: 14px;")
        slider_layout.addWidget(self.sensitivity_label)
        
        sensitivity_layout.addLayout(slider_layout)
        sensitivity_group.setLayout(sensitivity_layout)
        layout.addWidget(sensitivity_group)
        
        # 버튼들
        button_layout = QHBoxLayout()
        
        ok_btn = QPushButton("✅ 이 설정으로 적용")
        ok_btn.setStyleSheet("""
            QPushButton {
                background-color: #4CAF50; 
                color: white; 
                padding: 10px 20px; 
                font-weight: bold; 
                font-size: 12px;
                border-radius: 5px;
            }
            QPushButton:hover {
                background-color: #45a049;
            }
        """)
        ok_btn.clicked.connect(self.accept)
        button_layout.addWidget(ok_btn)
        
        cancel_btn = QPushButton("❌ 취소")
        cancel_btn.setStyleSheet("""
            QPushButton {
                background-color: #f44336; 
                color: white; 
                padding: 10px 20px; 
                font-weight: bold; 
                font-size: 12px;
                border-radius: 5px;
            }
            QPushButton:hover {
                background-color: #da190b;
            }
        """)
        cancel_btn.clicked.connect(self.reject)
        button_layout.addWidget(cancel_btn)
        
        layout.addLayout(button_layout)
        self.setLayout(layout)
    
    def detect_bubbles(self):
        """버블 감지 및 실시간 캔버스 업데이트"""
        try:
            x1, y1, x2, y2 = self.roi
            roi_image = self.image[y1:y2, x1:x2]
            
            params = {
                'sensitivity': self.sensitivity_slider.value(),
                'min_dist': 15,
                'min_radius': 4,
                'max_radius': 20
            }
            
            bubbles = self.detector.detect_bubbles(roi_image, params)
            
            # 원본 이미지 좌표로 변환
            self.detected_bubbles = [(x1 + x, y1 + y, r) for x, y, r in bubbles]
            
            # 🚀 실시간 캔버스 업데이트 - 이전 미리보기 마커들 제거 후 새로 표시
            self.canvas.clear_preview_markers()
            
            # 감지된 버블들을 실시간으로 캔버스에 표시
            preview_positions = [(x, y) for x, y, r in self.detected_bubbles]
            if preview_positions:
                self.canvas.add_preview_markers(preview_positions)
            
            # 결과 표시 업데이트
            count = len(self.detected_bubbles)
            if count == 0:
                self.result_label.setText("❌ 버블이 감지되지 않음")
                self.result_label.setStyleSheet("color: #f44336; font-size: 18px; font-weight: bold; padding: 15px;")
            else:
                self.result_label.setText(f"🎯 {count}개 버블 감지됨!")
                self.result_label.setStyleSheet("color: #4CAF50; font-size: 18px; font-weight: bold; padding: 15px;")
            
        except Exception as e:
            self.result_label.setText(f"⚠️ 오류: {str(e)}")
            self.result_label.setStyleSheet("color: #ff9800; font-size: 14px; padding: 15px;")
    
    def on_sensitivity_changed(self, value):
        """민감도 변경 시 실시간 업데이트"""
        self.sensitivity_label.setText(str(value))
        self.detect_bubbles()  # 실시간으로 감지 결과 업데이트
    
    def get_bubbles(self) -> List[Tuple[int, int]]:
        """감지된 버블 위치 반환"""
        return [(x, y) for x, y, r in self.detected_bubbles]
    
    def get_sensitivity(self) -> int:
        """현재 민감도 반환"""
        return self.sensitivity_slider.value()
    
    def closeEvent(self, event):
        """다이얼로그 닫힐 때 미리보기 마커들 정리"""
        self.canvas.clear_preview_markers()
        super().closeEvent(event)
    
    def reject(self):
        """취소 시 미리보기 마커들 정리"""
        self.canvas.clear_preview_markers()
        super().reject()

class CollapsibleGroupBox(QWidget):
    """접기/펼치기 가능한 그룹박스"""
    
    def __init__(self, title="", parent=None):
        super().__init__(parent)
        self.is_collapsed = True
        self.parent_modal = parent
        self.init_ui(title)
    
    def init_ui(self, title):
        self.main_layout = QVBoxLayout()
        self.main_layout.setContentsMargins(0, 0, 0, 0)
        self.main_layout.setSpacing(0)
        
        # 제목 버튼
        self.title_button = QPushButton(f"▶ {title}")
        self.title_button.setCheckable(True)
        self.title_button.setChecked(False)
        self.title_button.clicked.connect(self.toggle_collapse)
        self.title_button.setStyleSheet("""
            QPushButton {
                text-align: left;
                padding: 10px;
                font-weight: bold;
                font-size: 11px;
                border: 1px solid #5d6d7e;
                border-radius: 3px;
                background-color: #34495e;
                color: white;
            }
            QPushButton:hover {
                background-color: #3498db;
            }
            QPushButton:checked {
                background-color: #2980b9;
            }
        """)
        
        # 내용 영역
        self.content_area = QWidget()
        self.content_layout = QVBoxLayout()
        self.content_layout.setContentsMargins(5, 5, 5, 5)
        self.content_area.setLayout(self.content_layout)
        self.content_area.setStyleSheet("""
            QWidget {
                border: 1px solid #5d6d7e;
                border-top: none;
                border-radius: 0px 0px 3px 3px;
                background-color: white;
            }
            QPushButton {
                background-color: #f8f9fa;
                color: #2c3e50;
                border: 1px solid #dee2e6;
                border-radius: 3px;
                padding: 8px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #e9ecef;
                border-color: #3498db;
            }
            QPushButton:pressed {
                background-color: #dee2e6;
            }
            QSpinBox {
                background-color: #e74c3c;
                color: white;
                font-weight: bold;
                font-size: 12px;
                border: none;
                padding: 5px;
                border-radius: 3px;
            }
        """)
        
        self.content_area.hide()
        
        self.main_layout.addWidget(self.title_button)
        self.main_layout.addWidget(self.content_area)
        self.setLayout(self.main_layout)
    
    def toggle_collapse(self):
        if self.title_button.isChecked():
            if self.parent_modal:
                self.parent_modal.close_all_groups_except(self)
            
            self.content_area.show()
            self.title_button.setText(self.title_button.text().replace("▶", "▼"))
            self.is_collapsed = False
        else:
            self.content_area.hide()
            self.title_button.setText(self.title_button.text().replace("▼", "▶"))
            self.is_collapsed = True
    
    def close_group(self):
        if not self.is_collapsed:
            self.title_button.setChecked(False)
            self.content_area.hide()
            self.title_button.setText(self.title_button.text().replace("▼", "▶"))
            self.is_collapsed = True
    
    def add_widget(self, widget):
        self.content_layout.addWidget(widget)

class OMRSettingsPanel(QWidget):
    """OMR 설정 패널 (우측 패널에 통합)"""
    
    mode_changed = Signal(str, str, int)  # mode, question_num, sensitivity
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.current_question_num = 1
        self.accordion_groups = []
        self.selected_button = None  # 선택된 버튼 추적
        self.init_ui()
        
    def close_all_groups_except(self, except_group):
        for group in self.accordion_groups:
            if group != except_group:
                group.close_group()
    
    def init_ui(self):
        layout = QVBoxLayout()
        layout.setContentsMargins(5, 5, 5, 5)
        layout.setSpacing(3)
        
        # 타이틀
        title_label = QLabel("📋 OMR 영역 설정")
        title_label.setStyleSheet("font-size: 14px; font-weight: bold; padding: 5px; text-align: center; background-color: #e8f5e8;")
        title_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(title_label)
        
        # 스크롤 영역
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setStyleSheet("""
            QScrollArea {
                background-color: #f8f9fa;
                border: 1px solid #dee2e6;
                border-radius: 4px;
            }
            QScrollBar:vertical {
                background-color: #e9ecef;
                width: 12px;
                border-radius: 6px;
            }
            QScrollBar::handle:vertical {
                background-color: #6c757d;
                border-radius: 6px;
                min-height: 20px;
            }
            QScrollBar::handle:vertical:hover {
                background-color: #495057;
            }
        """)
        
        scroll_widget = QWidget()
        scroll_layout = QVBoxLayout()
        scroll_layout.setContentsMargins(5, 5, 5, 5)
        scroll_layout.setSpacing(3)
        
        self.create_accordion_groups(scroll_layout)
        scroll_layout.addStretch()
        
        scroll_widget.setLayout(scroll_layout)
        scroll.setWidget(scroll_widget)
        layout.addWidget(scroll)
        
        # 상태 영역 제거 (요청사항)
        
        self.setLayout(layout)
    
    def create_accordion_groups(self, layout):
        groups = [
            ("1. 학생정보", [
                ("계열", "계열", "1~9"),
                ("학년", "학년", "1~6년"),
                ("반(십의자리)", "반-10의자리", "0~9"),
                ("반(일의자리)", "반-1의자리", "0~9"),
                ("번호(십의자리)", "번호-10의자리", "0~9"),
                ("번호(일의자리)", "번호-1의자리", "0~9")
            ]),
            ("2. 결시코드", [("결시코드", "결시코드", "1~9")]),
            ("3. 과목코드", [
                ("과목코드(십의자리)", "과목코드-10의자리", "0~9"),
                ("과목코드(일의자리)", "과목코드-1의자리", "0~9")
            ]),
            ("4. 객관식", [("객관식 답안", "객관식", "답안 선택")], "objective"),
            ("5. 주관식총점", [
                ("총점(십의자리)", "주관식점수-10의자리", "0~9"),
                ("총점(일의자리)", "주관식점수-1의자리", "0~9"),
                ("총점(소수점)", "주관식점수-소수점첫자리", "0~9")
            ])
        ]
        
        self.accordion_groups.clear()
        
        for group_data in groups:
            if len(group_data) == 3:
                title, items, group_type = group_data
                group_widget = self.create_accordion_special_group(title, items, group_type)
            else:
                title, items = group_data
                group_widget = self.create_accordion_group(title, items)
            
            self.accordion_groups.append(group_widget)
            layout.addWidget(group_widget)
    
    def create_accordion_group(self, title, items):
        group = CollapsibleGroupBox(title, self)
        
        for text, mode, desc in items:
            btn = QPushButton(text)
            btn.setToolTip(desc)
            btn.setMinimumHeight(35)
            btn.clicked.connect(lambda checked, m=mode, t=text, b=btn: self.select_mode(m, t, b))
            group.add_widget(btn)
        
        return group
    
    def create_accordion_special_group(self, title, items, group_type):
        group = CollapsibleGroupBox(title, self)
        
        for text, mode, desc in items:
            item_widget = QWidget()
            item_layout = QHBoxLayout()
            item_layout.setContentsMargins(0, 0, 0, 0)
            item_layout.setSpacing(5)
            
            btn = QPushButton(text)
            btn.setToolTip(desc)
            btn.setMinimumHeight(35)
            btn.clicked.connect(lambda checked, m=mode, t=text, b=btn: self.select_mode(m, t, b))
            item_layout.addWidget(btn)
            
            if group_type == "objective":
                self.objective_spinbox = QSpinBox()
                self.objective_spinbox.setMinimum(1)
                self.objective_spinbox.setMaximum(999)
                self.objective_spinbox.setValue(self.current_question_num)
                self.objective_spinbox.setFixedWidth(60)
                self.objective_spinbox.setMinimumHeight(35)
                self.objective_spinbox.valueChanged.connect(self.update_objective_number)
                item_layout.addWidget(self.objective_spinbox)
            
            item_widget.setLayout(item_layout)
            group.add_widget(item_widget)
        
        return group
    
    def create_status_area(self, layout):
        # 제거됨 - 요청사항에 따라 상태 영역 삭제
        pass
    
    def select_mode(self, mode, display_text, button):
        # 이전 선택된 버튼 스타일 초기화
        if self.selected_button:
            self.selected_button.setStyleSheet("""
                QPushButton {
                    background-color: #f8f9fa;
                    color: #2c3e50;
                    border: 1px solid #dee2e6;
                    border-radius: 3px;
                    padding: 8px;
                    font-weight: bold;
                }
                QPushButton:hover {
                    background-color: #e9ecef;
                    border-color: #3498db;
                }
                QPushButton:pressed {
                    background-color: #dee2e6;
                }
            """)
        
        # 새로 선택된 버튼에 옅은 노란색 배경 적용
        button.setStyleSheet("""
            QPushButton {
                background-color: #fff9c4;
                color: #2c3e50;
                border: 2px solid #f59e0b;
                border-radius: 3px;
                padding: 8px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #fef3c7;
                border-color: #d97706;
            }
        """)
        
        self.selected_button = button
        
        if mode == "객관식":
            self.mode_changed.emit(mode, str(self.current_question_num), 60)
        else:
            self.mode_changed.emit(mode, "", 60)
    
    def update_objective_number(self, value):
        self.current_question_num = value
    
    def update_dynamic_buttons(self):
        if hasattr(self, 'objective_spinbox'):
            self.objective_spinbox.setValue(self.current_question_num)

class SimpleOMRSetup(QMainWindow):
    """단순화된 OMR 설정 프로그램"""
    
    def __init__(self):
        super().__init__()
        self.image_path = None
        self.original_image = None
        self.bubble_detector = BubbleDetector()
        self.rotation_corrector = ImageRotationCorrector()
        
        # 데이터 저장
        self.omr_areas = {}  # area_id: {'type': str, 'positions': [(x,y)], 'values': [str], 'sensitivity': int}
        self.area_counter = 0
        self.group_counter = 0
        
        # 현재 모드
        self.current_mode = ""
        self.current_question_num = ""
        self.current_sensitivity = 60
        
        self.init_folders()
        self.init_ui()
    
    def init_folders(self):
        """폴더 초기화"""
        os.makedirs("settings", exist_ok=True)
        os.makedirs("images", exist_ok=True)
    
    def init_ui(self):
        """UI 초기화"""
        self.setWindowTitle("OMR 설정 도구 v2.0 (개선된 버전)")
        self.setGeometry(100, 100, 1400, 800)
        
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        # 메인 레이아웃
        main_layout = QVBoxLayout()
        
        # 툴바
        toolbar = self.create_toolbar()
        main_layout.addWidget(toolbar)
        
        # 컨텐츠 영역 (3개 패널로 구성)
        content_splitter = QSplitter(Qt.Horizontal)
        
        # 좌측: 아코디언 설정 패널
        self.left_panel = self.create_left_panel()
        content_splitter.addWidget(self.left_panel)
        
        # 중앙: 이미지 영역 (메인)
        self.canvas = SimpleImageCanvas()
        self.canvas.area_selected.connect(self.on_area_selected)
        content_splitter.addWidget(self.canvas)
        
        # 우측: 설정된 영역 목록
        self.right_panel = self.create_right_panel()
        content_splitter.addWidget(self.right_panel)
        
        # 비율 설정: 좌측(1) : 중앙(4) : 우측(2)
        content_splitter.setSizes([250, 900, 400])
        content_splitter.setHandleWidth(4)
        
        # 패널 표시 상태 추적
        self.left_panel_visible = True
        self.right_panel_visible = True
        
        main_layout.addWidget(content_splitter)
        
        central_widget.setLayout(main_layout)
        
        # 스타일 적용
        self.setStyleSheet("""
            QMainWindow { background-color: #fafafa; }
            QPushButton { 
                padding: 8px 16px; 
                font-weight: bold; 
                border: none; 
                border-radius: 4px; 
                font-size: 11px;
            }
            QLabel { font-family: '맑은 고딕'; }
        """)
    
    def create_toolbar(self) -> QWidget:
        """툴바 생성"""
        toolbar = QFrame()
        toolbar.setFixedHeight(60)
        toolbar.setStyleSheet("background-color: #2196F3; padding: 8px;")
        
        layout = QHBoxLayout()
        
        # 좌측 버튼들
        load_btn = QPushButton("📁 이미지 불러오기")
        load_btn.setStyleSheet("background-color: white; color: #2196F3;")
        load_btn.clicked.connect(self.load_image)
        layout.addWidget(load_btn)
        
        correct_btn = QPushButton("📐 기울기 보정")
        correct_btn.setStyleSheet("background-color: #FF9800; color: white;")
        correct_btn.clicked.connect(self.correct_image_rotation)
        layout.addWidget(correct_btn)
        
        # 패널 토글 버튼들
        self.left_panel_btn = QPushButton("◀ 설정패널")
        self.left_panel_btn.setStyleSheet("background-color: #673AB7; color: white;")
        self.left_panel_btn.clicked.connect(self.toggle_left_panel)
        layout.addWidget(self.left_panel_btn)
        
        self.right_panel_btn = QPushButton("목록패널 ▶")
        self.right_panel_btn.setStyleSheet("background-color: #673AB7; color: white;")
        self.right_panel_btn.clicked.connect(self.toggle_right_panel)
        layout.addWidget(self.right_panel_btn)
        
        layout.addStretch()
        
        # 중앙 정보
        self.image_info = QLabel("이미지: 없음")
        self.image_info.setStyleSheet("color: white; font-weight: bold;")
        layout.addWidget(self.image_info)
        
        layout.addStretch()
        
        # 우측 버튼들
        save_btn = QPushButton("💾 설정 저장")
        save_btn.setStyleSheet("background-color: #4CAF50; color: white;")
        save_btn.clicked.connect(self.save_settings)
        layout.addWidget(save_btn)
        
        load_settings_btn = QPushButton("📂 설정 불러오기")
        load_settings_btn.setStyleSheet("background-color: #9C27B0; color: white;")
        load_settings_btn.clicked.connect(self.load_settings)
        layout.addWidget(load_settings_btn)
        
        clear_btn = QPushButton("🗑️ 초기화")
        clear_btn.setStyleSheet("background-color: #f44336; color: white;")
        clear_btn.clicked.connect(self.clear_all)
        layout.addWidget(clear_btn)
        
        toolbar.setLayout(layout)
        return toolbar
    
    def create_left_panel(self) -> QWidget:
        """좌측 패널 생성 (아코디언 설정)"""
        panel = QWidget()
        panel.setFixedWidth(250)  # 250px로 변경
        layout = QVBoxLayout()
        layout.setContentsMargins(5, 5, 5, 5)
        
        # OMR 설정 패널
        self.settings_panel = OMRSettingsPanel()
        self.settings_panel.mode_changed.connect(self.on_mode_changed)
        layout.addWidget(self.settings_panel)
        
        panel.setLayout(layout)
        return panel
    
    def create_right_panel(self) -> QWidget:
        """우측 패널 생성 (설정된 영역 목록)"""
        panel = QWidget()
        panel.setMinimumWidth(400)  # 최소 폭 설정
        layout = QVBoxLayout()
        layout.setContentsMargins(5, 5, 5, 5)
        
        # 설정된 영역 목록
        list_group = QGroupBox("📋 설정된 영역 목록")
        list_layout = QVBoxLayout()
        
        self.area_tree = QTreeWidget()
        self.area_tree.setHeaderLabels(['그룹', '영역', '문항', 'X', 'Y', '감도', '값'])
        
        # 컬럼 너비 설정 (더 넉넉하게)
        header = self.area_tree.header()
        header.resizeSection(0, 60)   # 그룹
        header.resizeSection(1, 120)  # 영역
        header.resizeSection(2, 60)   # 문항
        header.resizeSection(3, 50)   # X
        header.resizeSection(4, 50)   # Y
        header.resizeSection(5, 50)   # 감도
        header.resizeSection(6, 60)   # 값
        
        list_layout.addWidget(self.area_tree)
        
        # 삭제 버튼
        delete_btn = QPushButton("🗑️ 선택 항목 삭제")
        delete_btn.setStyleSheet("background-color: #f44336; color: white; padding: 10px;")
        delete_btn.clicked.connect(self.delete_selected_area)
        list_layout.addWidget(delete_btn)
        
        list_group.setLayout(list_layout)
        layout.addWidget(list_group)
        
        panel.setLayout(layout)
        return panel
    
    def toggle_left_panel(self):
        """좌측 패널 토글"""
        if self.left_panel_visible:
            self.left_panel.hide()
            self.left_panel_btn.setText("▶ 설정패널")
            self.left_panel_btn.setStyleSheet("background-color: #4CAF50; color: white;")
            self.left_panel_visible = False
        else:
            self.left_panel.show()
            self.left_panel_btn.setText("◀ 설정패널")
            self.left_panel_btn.setStyleSheet("background-color: #673AB7; color: white;")
            self.left_panel_visible = True
    
    def toggle_right_panel(self):
        """우측 패널 토글"""
        if self.right_panel_visible:
            self.right_panel.hide()
            self.right_panel_btn.setText("◀ 목록패널")
            self.right_panel_btn.setStyleSheet("background-color: #4CAF50; color: white;")
            self.right_panel_visible = False
        else:
            self.right_panel.show()
            self.right_panel_btn.setText("목록패널 ▶")
            self.right_panel_btn.setStyleSheet("background-color: #673AB7; color: white;")
            self.right_panel_visible = True
    
    def on_mode_changed(self, mode: str, question_num: str, sensitivity: int):
        """모드 변경 처리"""
        self.current_mode = mode
        self.current_question_num = question_num
        self.current_sensitivity = sensitivity
        
        # 상태바 제거로 메시지 표시 안함
    
    def load_image(self):
        """이미지 불러오기"""
        file_path, _ = QFileDialog.getOpenFileName(
            self, "OMR 이미지 선택", "",
            "이미지 파일 (*.jpg *.jpeg *.png *.bmp *.tiff)"
        )
        
        if file_path:
            self.image_path = file_path
            self.original_image = cv2.imread(file_path)
            
            if self.original_image is None:
                QMessageBox.critical(self, "오류", "이미지를 불러올 수 없습니다.")
                return
            
            success = self.canvas.set_image(file_path)
            if success:
                filename = os.path.basename(file_path)
                self.image_info.setText(f"이미지: {filename}")
                # 상태바 제거로 메시지 표시 안함
            else:
                QMessageBox.critical(self, "오류", "이미지 표시에 실패했습니다.")
    
    def correct_image_rotation(self):
        """이미지 기울기 보정"""
        if self.original_image is None:
            QMessageBox.warning(self, "알림", "먼저 이미지를 불러오세요.")
            return
        
        try:
            # 기울기 감지
            angle = self.rotation_corrector.detect_skew_angle(self.original_image)
            
            if abs(angle) < 0.5:
                QMessageBox.information(self, "보정 결과", "이미지가 이미 올바르게 정렬되어 있습니다.")
                return
            
            # 사용자 확인
            reply = QMessageBox.question(
                self, "기울기 보정", 
                f"감지된 기울기: {angle:.2f}°\n보정하시겠습니까?",
                QMessageBox.Yes | QMessageBox.No
            )
            
            if reply == QMessageBox.Yes:
                # 보정 적용
                corrected = self.rotation_corrector.correct_rotation(self.original_image, angle)
                
                # 임시 파일 저장
                temp_path = "temp_corrected.png"
                cv2.imwrite(temp_path, corrected)
                
                # 캔버스 업데이트
                self.original_image = corrected
                self.canvas.set_image(temp_path)
                
                # 상태바 제거로 메시지 표시 안함
                
                # 임시 파일 삭제
                if os.path.exists(temp_path):
                    os.remove(temp_path)
                    
        except Exception as e:
            QMessageBox.critical(self, "오류", f"보정 중 오류가 발생했습니다: {str(e)}")
    
    def on_area_selected(self, x1: int, y1: int, x2: int, y2: int):
        """영역 선택 처리"""
        if self.original_image is None:
            QMessageBox.warning(self, "알림", "먼저 이미지를 불러오세요.")
            return
        
        if not self.current_mode:
            QMessageBox.warning(self, "알림", "설정 패널에서 영역 유형을 선택하세요.")
            return
        
        # 버블 감지 다이얼로그 (캔버스 참조 전달)
        dialog = SimpleBubbleDialog(
            self.bubble_detector, 
            self.original_image, 
            (x1, y1, x2, y2),
            self.canvas,  # 🚀 캔버스 참조 전달로 실시간 미리보기 가능
            self
        )
        
        if dialog.exec() == QDialog.Accepted:
            bubbles = dialog.get_bubbles()
            sensitivity = dialog.get_sensitivity()
            
            if bubbles:
                # 영역 데이터 저장
                area_data = {
                    'type': self.current_mode,
                    'question_num': self.current_question_num,
                    'positions': bubbles,
                    'values': self.get_default_values(self.current_mode, len(bubbles)),
                    'sensitivity': sensitivity,
                    'group_id': self.group_counter
                }
                
                self.omr_areas[self.area_counter] = area_data
                self.area_counter += 1
                self.group_counter += 1
                
                # UI 업데이트
                self.update_area_list()
                # 미리보기 마커들을 확정 마커로 변환
                self.canvas.clear_preview_markers()
                self.canvas.add_markers(bubbles)
                
                bubble_count = len(bubbles)
                mode_display = f"{self.current_mode} {self.current_question_num}번" if self.current_question_num else self.current_mode
                # 상태바 제거로 메시지 표시 안함
                
                # 객관식의 경우 문항 번호 자동 증가
                if self.current_mode == "객관식":
                    self.settings_panel.current_question_num += 1
                    self.settings_panel.update_dynamic_buttons()
        else:
            # 취소 시 미리보기 마커들 정리 (dialog에서 자동 처리됨)
            pass
    
    def get_default_values(self, area_type: str, count: int) -> List[str]:
        """기본값 생성"""
        if area_type in ["학년", "계열", "결시코드"]:
            return [str(i + 1) for i in range(count)]
        elif "자리" in area_type or "소수점" in area_type:
            return [str(i) for i in range(count)]
        elif area_type == "객관식":
            return [str(i + 1) for i in range(count)]
        else:
            return [str(i + 1) for i in range(count)]
    
    def update_area_list(self):
        """영역 목록 업데이트"""
        self.area_tree.clear()
        
        for area_id, area_data in self.omr_areas.items():
            group_id = area_data.get('group_id', 0)
            
            for i, (x, y) in enumerate(area_data['positions']):
                item = QTreeWidgetItem([
                    f"G{group_id}",  # 그룹
                    area_data['type'],  # 영역
                    area_data.get('question_num', ''),  # 문항
                    str(x),  # X
                    str(y),  # Y
                    str(area_data.get('sensitivity', 60)),  # 감도
                    area_data['values'][i] if i < len(area_data['values']) else ''  # 값
                ])
                item.setData(0, Qt.UserRole, (area_id, i))
                self.area_tree.addTopLevelItem(item)
    
    def delete_selected_area(self):
        """선택된 영역 삭제"""
        current_item = self.area_tree.currentItem()
        if not current_item:
            QMessageBox.warning(self, "알림", "삭제할 항목을 선택하세요.")
            return
        
        area_id, pos_idx = current_item.data(0, Qt.UserRole)
        
        if area_id in self.omr_areas:
            # 특정 위치만 삭제할지, 전체 그룹을 삭제할지 묻기
            reply = QMessageBox.question(
                self, "삭제 확인", 
                "전체 그룹을 삭제하시겠습니까?\n(아니요를 선택하면 선택된 버블만 삭제됩니다)",
                QMessageBox.Yes | QMessageBox.No | QMessageBox.Cancel
            )
            
            if reply == QMessageBox.Yes:
                # 전체 그룹 삭제
                del self.omr_areas[area_id]
            elif reply == QMessageBox.No:
                # 선택된 버블만 삭제
                area_data = self.omr_areas[area_id]
                if len(area_data['positions']) > 1:
                    area_data['positions'].pop(pos_idx)
                    if pos_idx < len(area_data['values']):
                        area_data['values'].pop(pos_idx)
                else:
                    # 마지막 버블이면 전체 그룹 삭제
                    del self.omr_areas[area_id]
            else:
                return
            
            self.update_area_list()
            self.refresh_canvas()
            # 상태바 제거로 메시지 표시 안함
    
    def refresh_canvas(self):
        """캔버스 새로고침"""
        self.canvas.clear_markers()
        
        for area_data in self.omr_areas.values():
            self.canvas.add_markers(area_data['positions'])
    
    def clear_all(self):
        """모든 설정 초기화"""
        reply = QMessageBox.question(
            self, "초기화", 
            "모든 설정을 삭제하시겠습니까?",
            QMessageBox.Yes | QMessageBox.No
        )
        
        if reply == QMessageBox.Yes:
            self.omr_areas.clear()
            self.area_counter = 0
            self.group_counter = 0
            self.settings_panel.current_question_num = 1
            self.settings_panel.update_dynamic_buttons()
            
            self.update_area_list()
            self.canvas.clear_markers()
            # 상태바 제거로 메시지 표시 안함
    
    def save_settings(self):
        """설정 저장"""
        if not self.omr_areas:
            QMessageBox.warning(self, "알림", "저장할 설정이 없습니다.")
            return
        
        file_path, _ = QFileDialog.getSaveFileName(
            self, "설정 저장", "settings/omr_config.json",
            "JSON 파일 (*.json)"
        )
        
        if file_path:
            try:
                # JSON 직렬화를 위한 데이터 변환 함수
                def convert_to_serializable(obj):
                    """numpy 타입과 기타 직렬화 불가능한 타입을 Python 기본 타입으로 변환"""
                    if isinstance(obj, dict):
                        return {k: convert_to_serializable(v) for k, v in obj.items()}
                    elif isinstance(obj, list):
                        return [convert_to_serializable(v) for v in obj]
                    elif isinstance(obj, tuple):
                        return tuple(convert_to_serializable(v) for v in obj)
                    elif isinstance(obj, np.integer):
                        return int(obj)
                    elif isinstance(obj, np.floating):
                        return float(obj)
                    elif isinstance(obj, np.ndarray):
                        return obj.tolist()
                    elif hasattr(obj, 'item'):  # numpy scalar
                        return obj.item()
                    else:
                        return obj
                
                # 설정 데이터 직렬화 가능한 형태로 변환
                serializable_areas = convert_to_serializable(self.omr_areas)
                
                config = {
                    'version': '2.0',
                    'created_at': datetime.now().isoformat(),
                    'image_info': {
                        'path': self.image_path,
                        'size': convert_to_serializable(self.original_image.shape[:2]) if self.original_image is not None else None
                    },
                    'omr_areas': serializable_areas,
                    'counters': {
                        'area_counter': int(self.area_counter),
                        'group_counter': int(self.group_counter),
                        'current_question_num': int(self.settings_panel.current_question_num)
                    }
                }
                
                with open(file_path, 'w', encoding='utf-8') as f:
                    json.dump(config, f, ensure_ascii=False, indent=2)
                
                QMessageBox.information(self, "저장 완료", f"설정이 저장되었습니다.\n파일: {file_path}")
                # 상태바 제거로 메시지 표시 안함
                
            except Exception as e:
                QMessageBox.critical(self, "저장 오류", f"저장 중 오류가 발생했습니다: {str(e)}")
                print(f"저장 오류 상세: {e}")  # 디버깅용
    
    def load_settings(self):
        """설정 불러오기"""
        file_path, _ = QFileDialog.getOpenFileName(
            self, "설정 불러오기", "settings/",
            "JSON 파일 (*.json)"
        )
        
        if file_path:
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    config = json.load(f)
                
                # 데이터 검증
                if 'omr_areas' not in config:
                    QMessageBox.warning(self, "오류", "올바르지 않은 설정 파일입니다.")
                    return
                
                # 설정 적용
                self.omr_areas.clear()
                
                # 문자열 키를 정수로 변환
                for key, value in config['omr_areas'].items():
                    self.omr_areas[int(key)] = value
                
                # 카운터 복원
                counters = config.get('counters', {})
                self.area_counter = counters.get('area_counter', len(self.omr_areas))
                self.group_counter = counters.get('group_counter', 0)
                
                if counters.get('current_question_num'):
                    self.settings_panel.current_question_num = counters['current_question_num']
                    self.settings_panel.update_dynamic_buttons()
                
                self.update_area_list()
                self.refresh_canvas()
                
                count = len(self.omr_areas)
                QMessageBox.information(self, "불러오기 완료", f"{count}개 영역이 로드되었습니다.")
                # 상태바 제거로 메시지 표시 안함
                
            except Exception as e:
                QMessageBox.critical(self, "불러오기 오류", f"설정 불러오기 중 오류가 발생했습니다: {str(e)}")

def main():
    """메인 함수"""
    app = QApplication(sys.argv)
    
    # 애플리케이션 정보 설정
    app.setApplicationName("OMR 설정 도구 v2.0")
    app.setApplicationVersion("2.0")
    
    # 폰트 설정
    font = QFont("맑은 고딕", 9)
    app.setFont(font)
    
    try:
        window = SimpleOMRSetup()
        window.show()
        
        print("="*50)
        print("🎯 OMR 설정 도구 v2.0 (개선된 버전)")
        print("✨ 주요 개선사항:")
        print("  - 아코디언 스타일 설정 패널")
        print("  - 상세 영역 타입 분류")
        print("  - 향상된 데이터 표시")
        print("  - 실시간 버블 감지")
        print("="*50)
        
        sys.exit(app.exec())
        
    except ImportError as e:
        QMessageBox.critical(None, "패키지 오류",
                           f"필요한 패키지가 설치되지 않았습니다:\n{str(e)}\n\n"
                           "다음 명령어로 설치하세요:\n"
                           "pip install PySide6 opencv-python numpy")
        sys.exit(1)
    except Exception as e:
        QMessageBox.critical(None, "실행 오류", f"프로그램 실행 중 오류가 발생했습니다:\n{str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()