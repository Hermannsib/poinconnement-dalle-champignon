import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
from matplotlib.backends.backend_pdf import PdfPages
import tkinter as tk
from tkinter import ttk, scrolledtext, filedialog, messagebox
from dataclasses import dataclass
from typing import Tuple, List
import math
from datetime import datetime

# Couleurs du thème
COLORS = {
    'primary': '#2C3E50',
    'secondary': '#3498DB',
    'success': '#27AE60',
    'warning': '#F39C12',
    'danger': '#E74C3C',
    'light': '#ECF0F1',
    'dark': '#34495E',
    'white': '#FFFFFF',
    'info': '#5DADE2'
}

@dataclass
class MaterialProperties:
    """Propriétés des matériaux"""
    f_ck: float
    f_ck_cube: float
    f_ct: float
    E_c: float
    f_yk: float
    f_su: float
    E_s: float
    epsilon_su: float
    D_max: float
    
    @classmethod
    def create_from_concrete_class(cls, f_ck: float, D_max: float = 16.0):
        f_ct = 0.30 * f_ck**(2/3)
        E_c = 10000 * f_ck**(1/3)
        
        return cls(
            f_ck=f_ck,
            f_ck_cube=f_ck * 1.25,
            f_ct=f_ct,
            E_c=E_c,
            f_yk=500.0,
            f_su=550.0,
            E_s=205000.0,
            epsilon_su=0.05,
            D_max=D_max
        )

@dataclass
class SlabGeometry:
    """Géométrie de la dalle champignon"""
    h: float
    h_champignon: float
    r_champignon: float
    b_colonne: float
    h_colonne: float
    d_colonne: float
    column_type: str
    L_x: float
    L_y: float
    c: float = 30.0
    
    def get_d_moyen(self, at_champignon: bool = True) -> float:
        h_local = self.h_champignon if at_champignon else self.h
        return h_local - self.c - 25.0
    
    def get_rayon_dalle_equivalent(self) -> float:
        return 0.6 * min(self.L_x, self.L_y)
    
    def get_rayon_colonne_equivalent(self) -> float:
        """Rayon équivalent selon le type de colonne"""
        if self.column_type == 'circular':
            return self.d_colonne / 2
        elif self.column_type == 'square':
            return self.b_colonne / (2 * math.pi**0.5)
        else:
            return (self.b_colonne * self.h_colonne)**0.5 / (2 * math.pi**0.5)
    
    def get_perimetre_base(self) -> float:
        """Périmètre de base de la colonne"""
        if self.column_type == 'circular':
            return math.pi * self.d_colonne
        elif self.column_type == 'square':
            return 4 * self.b_colonne
        else:
            return 2 * (self.b_colonne + self.h_colonne)

@dataclass
class Reinforcement:
    """Armature de flexion"""
    rho_sup: float
    rho_inf: float
    phi_sup: float
    phi_inf: float
    
    def get_rho_sup_decimal(self) -> float:
        return self.rho_sup / 100.0
    
    def get_rho_inf_decimal(self) -> float:
        return self.rho_inf / 100.0
    
    def get_espacement(self, phi: float, rho: float, d: float) -> float:
        A_s_par_metre = rho * d * 1000 / 100
        A_barre = math.pi * (phi/2)**2
        nombre_barres = A_s_par_metre / A_barre
        return 1000 / nombre_barres if nombre_barres > 0 else 200

class PunchingShearCalculator:
    """Calculateur de résistance au poinçonnement selon Muttoni"""
    
    def __init__(self, materials: MaterialProperties, geometry: SlabGeometry, 
                 reinforcement: Reinforcement, gamma_c: float = 1.5, gamma_s: float = 1.15):
        self.materials = materials
        self.geometry = geometry
        self.reinforcement = reinforcement
        self.gamma_c = gamma_c
        self.gamma_s = gamma_s
        
        self.f_cd = self.calculate_f_cd()
        self.f_yd = self.materials.f_yk / gamma_s
        self.tau_cd = self.calculate_tau_cd()
        
    def calculate_f_cd(self) -> float:
        eta_fc = min(1.0, 1.0 - self.materials.f_ck / 250.0)
        return eta_fc * self.materials.f_ck / self.gamma_c
    
    def calculate_tau_cd(self) -> float:
        return 0.2 * self.materials.f_ck
    
    def calculate_k_Dmax(self) -> float:
        return min(1.0, 48.0 / (16.0 + self.materials.D_max))
    
    def get_perimetre_controle(self, at_champignon: bool = True) -> float:
        """Périmètre de contrôle à d/2 du bord de la colonne"""
        d = self.geometry.get_d_moyen(at_champignon)
        geom = self.geometry
        
        if geom.column_type == 'circular':
            u = math.pi * (geom.d_colonne + d)
        elif geom.column_type == 'square':
            u = 4 * geom.b_colonne + math.pi * d
        else:
            u = 2 * (geom.b_colonne + geom.h_colonne) + math.pi * d
        
        return u
    
    def calculate_tau_Rd(self, psi: float, at_champignon: bool = True) -> float:
        d = self.geometry.get_d_moyen(at_champignon)
        k_Dmax = self.calculate_k_Dmax()
        
        denominator = 1.5 * (d/1000) * psi * k_Dmax + 0.135
        tau_Rd = self.tau_cd * 0.45 / denominator
        
        return tau_Rd
    
    def calculate_V_Rd(self, psi: float, at_champignon: bool = True) -> float:
        tau_Rd = self.calculate_tau_Rd(psi, at_champignon)
        u = self.get_perimetre_controle(at_champignon)
        d = self.geometry.get_d_moyen(at_champignon)
        
        V_Rd = tau_Rd * u * d / 1000
        return V_Rd
    
    def calculate_moment_curvature_relation(self, at_champignon: bool = True) -> Tuple[np.ndarray, np.ndarray]:
        d = self.geometry.get_d_moyen(at_champignon)
        rho = self.reinforcement.get_rho_sup_decimal()
        
        m_r = self.materials.f_ct * (self.geometry.h**2 / 6) / 1e6
        kappa_r = self.materials.f_ct / (self.materials.E_c * d/2) * 1000
        
        m_pl = rho * d * self.f_yd * (d - 0.4 * rho * d * self.f_yd / self.f_cd) / 1e3
        
        EI_fissure = self.materials.E_s * rho * d**3 * (1 - 0.4 * rho) / 1e9
        
        kappa_values = []
        m_values = []
        
        kappa_values.extend([0, kappa_r])
        m_values.extend([0, m_r])
        
        if m_pl > m_r:
            kappa_y = m_pl / EI_fissure
            kappa_values.append(kappa_y)
            m_values.append(m_pl)
            
            kappa_values.extend([kappa_y * 5, kappa_y * 10])
            m_values.extend([m_pl, m_pl])
        
        return np.array(kappa_values), np.array(m_values)
    
    def calculate_load_rotation_curve(self, at_champignon: bool = True, 
                                     n_points: int = 50) -> Tuple[np.ndarray, np.ndarray]:
        d = self.geometry.get_d_moyen(at_champignon)
        r_b = self.geometry.get_rayon_dalle_equivalent()
        r_a = self.geometry.get_rayon_colonne_equivalent()
        
        kappa, m = self.calculate_moment_curvature_relation(at_champignon)
        
        factor = 2 * math.pi / math.log(max(r_b/r_a, 2.0))
        
        V_values = m * factor
        psi_values = kappa * (r_b - r_a) / 2000
        
        return psi_values, V_values
    
    def find_punching_resistance(self, at_champignon: bool = True) -> Tuple[float, float, float]:
        psi_values, V_curve = self.calculate_load_rotation_curve(at_champignon)
        
        V_criterion = []
        for psi in psi_values:
            if psi > 0:
                V_crit = self.calculate_V_Rd(psi, at_champignon)
                V_criterion.append(V_crit)
            else:
                V_criterion.append(1e6)
        
        V_criterion = np.array(V_criterion)
        V_Rd_array = np.minimum(V_curve, V_criterion)
        
        idx_max = np.argmax(V_Rd_array)
        V_Rd = V_Rd_array[idx_max]
        psi_Rd = psi_values[idx_max]
        
        tau_Rd = self.calculate_tau_Rd(psi_Rd, at_champignon)
        ratio = tau_Rd / self.tau_cd if self.tau_cd > 0 else 0
        
        return V_Rd, psi_Rd, ratio

class PunchingShearGUI:
    """Interface graphique améliorée"""
    
    def __init__(self, root):
        self.root = root
        self.root.title("Vérification au Poinçonnement - Dalle Champignon (Méthode Muttoni)")
        
        self.root.state('zoomed')
        
        self.setup_styles()
        
        self.main_frame = tk.Frame(root, bg=COLORS['light'])
        self.main_frame.pack(fill='both', expand=True)
        
        title_frame = tk.Frame(self.main_frame, bg=COLORS['primary'], height=80)
        title_frame.pack(fill='x', side='top')
        title_frame.pack_propagate(False)
        
        title_label = tk.Label(title_frame, 
                              text="🏗️ VÉRIFICATION AU POINÇONNEMENT - DALLE CHAMPIGNON",
                              font=('Arial', 20, 'bold'),
                              bg=COLORS['primary'],
                              fg=COLORS['white'])
        title_label.pack(expand=True)
        
        subtitle_label = tk.Label(title_frame,
                                 text="Méthode Muttoni ",
                                 font=('Arial', 11),
                                 bg=COLORS['primary'],
                                 fg=COLORS['info'])
        subtitle_label.pack()
        
        self.notebook = ttk.Notebook(self.main_frame)
        self.notebook.pack(fill='both', expand=True, padx=10, pady=10)
        
        self.frame_input = tk.Frame(self.notebook, bg=COLORS['white'])
        self.notebook.add(self.frame_input, text='📊 Données d\'entrée')
        
        self.frame_results = tk.Frame(self.notebook, bg=COLORS['white'])
        self.notebook.add(self.frame_results, text='📋 Note de Calcul')
        
        self.frame_graphs = tk.Frame(self.notebook, bg=COLORS['white'])
        self.notebook.add(self.frame_graphs, text='📈 Graphiques')
        
        self.frame_reinforcement = tk.Frame(self.notebook, bg=COLORS['white'])
        self.notebook.add(self.frame_reinforcement, text='🔧 Plans d\'Armature')
        
        self.init_variables()
        self.create_input_interface()
        self.create_results_interface()
        self.create_graphs_interface()
        self.create_reinforcement_interface()
        
    def setup_styles(self):
        """Configure les styles ttk"""
        style = ttk.Style()
        style.theme_use('clam')
        
        style.configure('TNotebook', background=COLORS['light'])
        style.configure('TNotebook.Tab', 
                       background=COLORS['secondary'],
                       foreground=COLORS['white'],
                       padding=[20, 10],
                       font=('Arial', 10, 'bold'))
        style.map('TNotebook.Tab',
                 background=[('selected', COLORS['primary'])],
                 foreground=[('selected', COLORS['white'])])
        
        style.configure('Action.TButton',
                       background=COLORS['success'],
                       foreground=COLORS['white'],
                       font=('Arial', 11, 'bold'),
                       padding=10)
        
        style.configure('Export.TButton',
                       background=COLORS['secondary'],
                       foreground=COLORS['white'],
                       font=('Arial', 10, 'bold'),
                       padding=8)
        
    def init_variables(self):
        """Initialise les variables"""
        self.var_fck = tk.DoubleVar(value=30.0)
        self.var_Dmax = tk.DoubleVar(value=16.0)
        self.var_fyk = tk.DoubleVar(value=500.0)
        
        self.var_h = tk.DoubleVar(value=200.0)
        self.var_h_champ = tk.DoubleVar(value=350.0)
        self.var_r_champ = tk.DoubleVar(value=800.0)
        self.var_Lx = tk.DoubleVar(value=8000.0)
        self.var_Ly = tk.DoubleVar(value=8000.0)
        self.var_c = tk.DoubleVar(value=30.0)
        
        self.var_column_type = tk.StringVar(value='circular')
        self.var_d_colonne = tk.DoubleVar(value=400.0)
        self.var_b_col = tk.DoubleVar(value=300.0)
        self.var_h_col = tk.DoubleVar(value=300.0)
        
        self.var_rho_sup = tk.DoubleVar(value=0.8)
        self.var_rho_inf = tk.DoubleVar(value=0.4)
        self.var_phi_sup = tk.DoubleVar(value=16.0)
        self.var_phi_inf = tk.DoubleVar(value=16.0)
        
        self.var_V_Ed = tk.DoubleVar(value=1000.0)
        
    def create_input_interface(self):
        """Interface de saisie améliorée"""
        canvas = tk.Canvas(self.frame_input, bg=COLORS['white'])
        scrollbar = ttk.Scrollbar(self.frame_input, orient="vertical", command=canvas.yview)
        scrollable_frame = tk.Frame(canvas, bg=COLORS['white'])
        
        scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )
        
        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        
        frame_col_type = tk.LabelFrame(scrollable_frame, 
                                       text="🔷 TYPE DE COLONNE",
                                       font=('Arial', 11, 'bold'),
                                       bg=COLORS['white'],
                                       fg=COLORS['primary'],
                                       padx=15, pady=15)
        frame_col_type.grid(row=0, column=0, padx=15, pady=10, sticky='ew')
        
        tk.Label(frame_col_type, text="Sélectionnez le type de colonne:",
                font=('Arial', 10, 'bold'), bg=COLORS['white']).grid(row=0, column=0, columnspan=3, sticky='w', pady=5)
        
        self.radio_circular = tk.Radiobutton(frame_col_type, 
                                            text="⭕ Circulaire", 
                                            variable=self.var_column_type,
                                            value='circular',
                                            command=self.update_column_inputs,
                                            font=('Arial', 10),
                                            bg=COLORS['white'],
                                            activebackground=COLORS['light'])
        self.radio_circular.grid(row=1, column=0, sticky='w', padx=10, pady=5)
        
        self.radio_square = tk.Radiobutton(frame_col_type,
                                          text="⬜ Carrée",
                                          variable=self.var_column_type,
                                          value='square',
                                          command=self.update_column_inputs,
                                          font=('Arial', 10),
                                          bg=COLORS['white'],
                                          activebackground=COLORS['light'])
        self.radio_square.grid(row=1, column=1, sticky='w', padx=10, pady=5)
        
        self.radio_rect = tk.Radiobutton(frame_col_type,
                                        text="▭ Rectangulaire",
                                        variable=self.var_column_type,
                                        value='rectangular',
                                        command=self.update_column_inputs,
                                        font=('Arial', 10),
                                        bg=COLORS['white'],
                                        activebackground=COLORS['light'])
        self.radio_rect.grid(row=1, column=2, sticky='w', padx=10, pady=5)
        
        self.frame_col_dims = tk.Frame(frame_col_type, bg=COLORS['white'])
        self.frame_col_dims.grid(row=2, column=0, columnspan=3, sticky='ew', pady=10)
        
        self.create_column_dimension_fields()
        
        frame_mat = tk.LabelFrame(scrollable_frame, 
                                 text="🧱 PROPRIÉTÉS DES MATÉRIAUX",
                                 font=('Arial', 11, 'bold'),
                                 bg=COLORS['white'],
                                 fg=COLORS['primary'],
                                 padx=15, pady=15)
        frame_mat.grid(row=1, column=0, padx=15, pady=10, sticky='ew')
        
        self.create_labeled_entry(frame_mat, "Résistance caractéristique du béton f_ck (MPa):", 
                                 self.var_fck, 0, COLORS['secondary'])
        self.create_labeled_entry(frame_mat, "Diamètre maximal granulat D_max (mm):", 
                                 self.var_Dmax, 1, COLORS['secondary'])
        self.create_labeled_entry(frame_mat, "Limite d'élasticité acier f_yk (MPa):", 
                                 self.var_fyk, 2, COLORS['secondary'])
        
        frame_geom = tk.LabelFrame(scrollable_frame, 
                                  text="📐 GÉOMÉTRIE DE LA DALLE CHAMPIGNON",
                                  font=('Arial', 11, 'bold'),
                                  bg=COLORS['white'],
                                  fg=COLORS['primary'],
                                  padx=15, pady=15)
        frame_geom.grid(row=2, column=0, padx=15, pady=10, sticky='ew')
        
        self.create_labeled_entry(frame_geom, "Épaisseur dalle courante h (mm):", 
                                 self.var_h, 0, COLORS['info'])
        self.create_labeled_entry(frame_geom, "Épaisseur champignon h_champ (mm):", 
                                 self.var_h_champ, 1, COLORS['info'])
        self.create_labeled_entry(frame_geom, "Rayon champignon r_champ (mm):", 
                                 self.var_r_champ, 2, COLORS['info'])
        self.create_labeled_entry(frame_geom, "Portée L_x (mm):", 
                                 self.var_Lx, 3, COLORS['info'])
        self.create_labeled_entry(frame_geom, "Portée L_y (mm):", 
                                 self.var_Ly, 4, COLORS['info'])
        self.create_labeled_entry(frame_geom, "Enrobage c (mm):", 
                                 self.var_c, 5, COLORS['info'])
        
        frame_reinf = tk.LabelFrame(scrollable_frame, 
                                   text="🔩 ARMATURES DE FLEXION",
                                   font=('Arial', 11, 'bold'),
                                   bg=COLORS['white'],
                                   fg=COLORS['primary'],
                                   padx=15, pady=15)
        frame_reinf.grid(row=3, column=0, padx=15, pady=10, sticky='ew')
        
        self.create_labeled_entry(frame_reinf, "Taux d'armature supérieur ρ_sup (%):", 
                                 self.var_rho_sup, 0, COLORS['success'])
        self.create_labeled_entry(frame_reinf, "Taux d'armature inférieur ρ_inf (%):", 
                                 self.var_rho_inf, 1, COLORS['success'])
        self.create_labeled_entry(frame_reinf, "Diamètre barres supérieures Ø_sup (mm):", 
                                 self.var_phi_sup, 2, COLORS['success'])
        self.create_labeled_entry(frame_reinf, "Diamètre barres inférieures Ø_inf (mm):", 
                                 self.var_phi_inf, 3, COLORS['success'])
        
        frame_load = tk.LabelFrame(scrollable_frame, 
                                  text="⚡ SOLLICITATIONS",
                                  font=('Arial', 11, 'bold'),
                                  bg=COLORS['white'],
                                  fg=COLORS['primary'],
                                  padx=15, pady=15)
        frame_load.grid(row=4, column=0, padx=15, pady=10, sticky='ew')
        
        self.create_labeled_entry(frame_load, "Effort tranchant de calcul V_Ed (kN):", 
                                 self.var_V_Ed, 0, COLORS['danger'])
        
        btn_frame = tk.Frame(scrollable_frame, bg=COLORS['white'])
        btn_frame.grid(row=5, column=0, pady=30)
        
        btn_calculate = ttk.Button(btn_frame, 
                                  text="🧮 CALCULER",
                                  command=self.perform_calculation,
                                  style='Action.TButton',
                                  width=30)
        btn_calculate.pack()
        
        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")
        
        self.update_column_inputs()
        
    def create_column_dimension_fields(self):
        """Crée les champs pour les dimensions de colonne"""
        self.label_d = tk.Label(self.frame_col_dims, 
                               text="Diamètre colonne d (mm):",
                               font=('Arial', 10),
                               bg=COLORS['white'])
        self.entry_d = tk.Entry(self.frame_col_dims, 
                               textvariable=self.var_d_colonne,
                               font=('Arial', 10),
                               width=15,
                               bd=2,
                               relief='solid')
        
        self.label_b = tk.Label(self.frame_col_dims,
                               text="Dimension colonne b (mm):",
                               font=('Arial', 10),
                               bg=COLORS['white'])
        self.entry_b = tk.Entry(self.frame_col_dims,
                               textvariable=self.var_b_col,
                               font=('Arial', 10),
                               width=15,
                               bd=2,
                               relief='solid')
        
        self.label_b_rect = tk.Label(self.frame_col_dims,
                                    text="Largeur colonne b (mm):",
                                    font=('Arial', 10),
                                    bg=COLORS['white'])
        self.entry_b_rect = tk.Entry(self.frame_col_dims,
                                     textvariable=self.var_b_col,
                                     font=('Arial', 10),
                                     width=15,
                                     bd=2,
                                     relief='solid')
        
        self.label_h_rect = tk.Label(self.frame_col_dims,
                                    text="Hauteur colonne h (mm):",
                                    font=('Arial', 10),
                                    bg=COLORS['white'])
        self.entry_h_rect = tk.Entry(self.frame_col_dims,
                                     textvariable=self.var_h_col,
                                     font=('Arial', 10),
                                     width=15,
                                     bd=2,
                                     relief='solid')
        
    def update_column_inputs(self):
        """Met à jour l'affichage des champs selon le type de colonne"""
        for widget in self.frame_col_dims.winfo_children():
            widget.grid_forget()
        
        col_type = self.var_column_type.get()
        
        if col_type == 'circular':
            self.label_d.grid(row=0, column=0, sticky='w', pady=5)
            self.entry_d.grid(row=0, column=1, padx=10, pady=5)
        elif col_type == 'square':
            self.label_b.grid(row=0, column=0, sticky='w', pady=5)
            self.entry_b.grid(row=0, column=1, padx=10, pady=5)
        else:
            self.label_b_rect.grid(row=0, column=0, sticky='w', pady=5)
            self.entry_b_rect.grid(row=0, column=1, padx=10, pady=5)
            self.label_h_rect.grid(row=1, column=0, sticky='w', pady=5)
            self.entry_h_rect.grid(row=1, column=1, padx=10, pady=5)
    
    def create_labeled_entry(self, parent, label_text, variable, row, color):
        """Crée un champ de saisie avec label coloré"""
        label = tk.Label(parent, 
                        text=label_text,
                        font=('Arial', 10),
                        bg=COLORS['white'],
                        fg=COLORS['dark'])
        label.grid(row=row, column=0, sticky='w', pady=5)
        
        entry = tk.Entry(parent, 
                        textvariable=variable,
                        font=('Arial', 10),
                        width=15,
                        bd=2,
                        relief='solid')
        entry.grid(row=row, column=1, padx=10, pady=5)
        
        indicator = tk.Label(parent, text="●", font=('Arial', 16), fg=color, bg=COLORS['white'])
        indicator.grid(row=row, column=2, padx=5)
    
    def create_results_interface(self):
        """Interface de résultats avec couleurs"""
        btn_frame = tk.Frame(self.frame_results, bg=COLORS['white'])
        btn_frame.pack(fill='x', padx=10, pady=10)
        
        ttk.Button(btn_frame, 
                  text="💾 Exporter Note de Calcul (PDF)",
                  command=self.export_report_pdf,
                  style='Export.TButton').pack(side='left', padx=5)
        
        ttk.Button(btn_frame,
                  text="📊 Exporter Graphiques (PDF)",
                  command=self.export_graphs_pdf,
                  style='Export.TButton').pack(side='left', padx=5)
        
        ttk.Button(btn_frame,
                  text="📦 Exporter Tout (PDF)",
                  command=self.export_all_pdf,
                  style='Export.TButton').pack(side='left', padx=5)
        
        # Zone de texte en Times New Roman avec texte noir
        self.text_results = tk.Text(self.frame_results, 
                                   width=120, 
                                   height=35,
                                   font=('Times New Roman', 10),
                                   bg=COLORS['white'],
                                   fg='black',
                                   wrap='word')
        
        # Configuration des tags - tous en noir
        self.text_results.tag_configure('normal', foreground='black', font=('Times New Roman', 10))
        self.text_results.tag_configure('bold', foreground='black', font=('Times New Roman', 10, 'bold'))
        self.text_results.tag_configure('header', foreground='black', font=('Times New Roman', 12, 'bold'))
        self.text_results.tag_configure('subheader', foreground='black', font=('Times New Roman', 11, 'bold'))
        self.text_results.tag_configure('success', foreground=COLORS['success'], font=('Times New Roman', 10, 'bold'))
        self.text_results.tag_configure('danger', foreground=COLORS['danger'], font=('Times New Roman', 10, 'bold'))
        self.text_results.tag_configure('warning', foreground=COLORS['warning'], font=('Times New Roman', 10, 'bold'))
        
        scrollbar = ttk.Scrollbar(self.frame_results, command=self.text_results.yview)
        self.text_results.configure(yscrollcommand=scrollbar.set)
        
        self.text_results.pack(side='left', fill='both', expand=True, padx=(10, 0), pady=10)
        scrollbar.pack(side='right', fill='y', pady=10, padx=(0, 10))
    
    def create_graphs_interface(self):
        """Interface pour les graphiques"""
        btn_frame = tk.Frame(self.frame_graphs, bg=COLORS['white'])
        btn_frame.pack(side='top', fill='x', padx=10, pady=10)
        
        ttk.Button(btn_frame, 
                  text="🔄 Rafraîchir",
                  command=self.update_graphs,
                  style='Export.TButton').pack(side='left', padx=5)
        
        self.frame_plot = tk.Frame(self.frame_graphs, bg=COLORS['white'])
        self.frame_plot.pack(fill='both', expand=True)
    
    def create_reinforcement_interface(self):
        """Interface pour les plans d'armature"""
        btn_frame = tk.Frame(self.frame_reinforcement, bg=COLORS['white'])
        btn_frame.pack(side='top', fill='x', padx=10, pady=10)
        
        ttk.Button(btn_frame,
                  text="📋 Exporter Plans (PDF)",
                  command=self.export_reinforcement_pdf,
                  style='Export.TButton').pack(side='left', padx=5)
        
        self.frame_reinf_canvas = tk.Frame(self.frame_reinforcement, bg=COLORS['white'])
        self.frame_reinf_canvas.pack(fill='both', expand=True)
    
    def perform_calculation(self):
        """Effectue les calculs"""
        try:
            materials = MaterialProperties.create_from_concrete_class(
                self.var_fck.get(), 
                self.var_Dmax.get()
            )
            materials.f_yk = self.var_fyk.get()
            
            col_type = self.var_column_type.get()
            
            geometry = SlabGeometry(
                h=self.var_h.get(),
                h_champignon=self.var_h_champ.get(),
                r_champignon=self.var_r_champ.get(),
                b_colonne=self.var_b_col.get() if col_type != 'circular' else 0,
                h_colonne=self.var_h_col.get() if col_type == 'rectangular' else self.var_b_col.get(),
                d_colonne=self.var_d_colonne.get() if col_type == 'circular' else 0,
                column_type=col_type,
                L_x=self.var_Lx.get(),
                L_y=self.var_Ly.get(),
                c=self.var_c.get()
            )
            
            reinforcement = Reinforcement(
                rho_sup=self.var_rho_sup.get(),
                rho_inf=self.var_rho_inf.get(),
                phi_sup=self.var_phi_sup.get(),
                phi_inf=self.var_phi_inf.get()
            )
            
            self.calculator = PunchingShearCalculator(materials, geometry, reinforcement)
            
            V_Rd_champ, psi_Rd_champ, ratio_champ = self.calculator.find_punching_resistance(at_champignon=True)
            V_Rd_dalle, psi_Rd_dalle, ratio_dalle = self.calculator.find_punching_resistance(at_champignon=False)
            
            self.generate_detailed_report(V_Rd_champ, psi_Rd_champ, ratio_champ,
                                         V_Rd_dalle, psi_Rd_dalle, ratio_dalle)
            
            self.update_graphs()
            self.draw_reinforcement_plan()
            
            messagebox.showinfo("Succès", "Calculs effectués avec succès!")
            
        except Exception as e:
            messagebox.showerror("Erreur", f"Erreur lors du calcul:\n{str(e)}")
            import traceback
            traceback.print_exc()
    
    def generate_detailed_report(self, V_Rd_champ, psi_Rd_champ, ratio_champ,
                                V_Rd_dalle, psi_Rd_dalle, ratio_dalle):
        """Génère le rapport détaillé avec formules"""
        self.text_results.delete(1.0, tk.END)
        calc = self.calculator
        
        # En-tête
        self.insert_text("="*100 + "\n", 'header')
        self.insert_text("          VÉRIFICATION AU POINÇONNEMENT DE DALLE CHAMPIGNON\n", 'header')
        self.insert_text("               Méthode Muttoni (Critère de rupture)\n", 'header')
        self.insert_text("                    Conforme à la thèse EPFL 3380\n", 'normal')
        self.insert_text("="*100 + "\n\n", 'header')
        
        now = datetime.now()
        self.insert_text(f"Date du calcul: {now.strftime('%d/%m/%Y %H:%M:%S')}\n\n", 'normal')
        
        # 1. DONNÉES D'ENTRÉE
        self.insert_text("1. DONNÉES D'ENTRÉE\n", 'header')
        self.insert_text("─"*100 + "\n\n", 'normal')
        
        self.insert_text("1.1 TYPE DE COLONNE\n", 'subheader')
        col_type_names = {'circular': 'Circulaire', 'square': 'Carrée', 'rectangular': 'Rectangulaire'}
        self.insert_text(f"   Type de colonne: {col_type_names[calc.geometry.column_type]}\n", 'normal')
        
        if calc.geometry.column_type == 'circular':
            self.insert_text(f"   Diamètre: d = {calc.geometry.d_colonne:.0f} mm\n", 'normal')
            self.insert_text(f"   Périmètre de base: u₀ = π × d = π × {calc.geometry.d_colonne:.0f} = {calc.geometry.get_perimetre_base():.0f} mm\n", 'normal')
        elif calc.geometry.column_type == 'square':
            self.insert_text(f"   Dimension: b = {calc.geometry.b_colonne:.0f} mm\n", 'normal')
            self.insert_text(f"   Périmètre de base: u₀ = 4 × b = 4 × {calc.geometry.b_colonne:.0f} = {calc.geometry.get_perimetre_base():.0f} mm\n", 'normal')
        else:
            self.insert_text(f"   Largeur: b = {calc.geometry.b_colonne:.0f} mm\n", 'normal')
            self.insert_text(f"   Hauteur: h = {calc.geometry.h_colonne:.0f} mm\n", 'normal')
            self.insert_text(f"   Périmètre de base: u₀ = 2(b + h) = 2({calc.geometry.b_colonne:.0f} + {calc.geometry.h_colonne:.0f}) = {calc.geometry.get_perimetre_base():.0f} mm\n", 'normal')
        
        r_eq = calc.geometry.get_rayon_colonne_equivalent()
        if calc.geometry.column_type == 'circular':
            self.insert_text(f"   Rayon équivalent: r_a,eq = d/2 = {calc.geometry.d_colonne:.0f}/2 = {r_eq:.1f} mm\n\n", 'normal')
        elif calc.geometry.column_type == 'square':
            self.insert_text(f"   Rayon équivalent: r_a,eq = b/(2√π) = {calc.geometry.b_colonne:.0f}/(2×√π) = {r_eq:.1f} mm\n\n", 'normal')
        else:
            self.insert_text(f"   Rayon équivalent: r_a,eq = √(b×h)/(2√π) = √({calc.geometry.b_colonne:.0f}×{calc.geometry.h_colonne:.0f})/(2×√π) = {r_eq:.1f} mm\n\n", 'normal')
        
        self.insert_text("1.2 MATÉRIAUX\n", 'subheader')
        self.insert_text("   • Béton:\n", 'bold')
        self.insert_text(f"     - Résistance caractéristique                f_ck    = {calc.materials.f_ck:.1f} MPa\n", 'normal')
        
        self.insert_text(f"     - Résistance à la traction (formule):     f_ct    = 0.30 × f_ck^(2/3)\n", 'normal')
        self.insert_text(f"                                               f_ct    = 0.30 × {calc.materials.f_ck:.1f}^(2/3) = {calc.materials.f_ct:.2f} MPa\n", 'normal')
        
        self.insert_text(f"     - Module d'élasticité (formule):          E_c     = 10000 × f_ck^(1/3)\n", 'normal')
        self.insert_text(f"                                               E_c     = 10000 × {calc.materials.f_ck:.1f}^(1/3) = {calc.materials.E_c:.0f} MPa\n", 'normal')
        
        self.insert_text(f"     - Diamètre maximal du granulat            D_max   = {calc.materials.D_max:.0f} mm\n", 'normal')
        
        k_Dmax = calc.calculate_k_Dmax()
        self.insert_text(f"     - Coefficient granulat (formule):         k_Dmax  = min(1.0, 48/(16 + D_max))\n", 'normal')
        self.insert_text(f"                                               k_Dmax  = min(1.0, 48/(16 + {calc.materials.D_max:.0f}))\n", 'normal')
        self.insert_text(f"                                               k_Dmax  = {k_Dmax:.3f}\n\n", 'normal')
        
        self.insert_text("   • Acier:\n", 'bold')
        self.insert_text(f"     - Limite d'élasticité caractéristique     f_yk    = {calc.materials.f_yk:.0f} MPa\n", 'normal')
        self.insert_text(f"     - Coefficient de sécurité                 γ_s     = {calc.gamma_s:.2f}\n", 'normal')
        self.insert_text(f"     - Limite d'élasticité de calcul:         f_yd    = f_yk / γ_s\n", 'normal')
        self.insert_text(f"                                               f_yd    = {calc.materials.f_yk:.0f} / {calc.gamma_s:.2f} = {calc.f_yd:.1f} MPa\n\n", 'normal')
        
        self.insert_text("1.3 GÉOMÉTRIE\n", 'subheader')
        self.insert_text(f"   • Épaisseur courante                       h       = {calc.geometry.h:.0f} mm\n", 'normal')
        self.insert_text(f"   • Épaisseur champignon                     h_champ = {calc.geometry.h_champignon:.0f} mm\n", 'normal')
        self.insert_text(f"   • Rayon champignon                         r_champ = {calc.geometry.r_champignon:.0f} mm\n", 'normal')
        self.insert_text(f"   • Enrobage                                 c       = {calc.geometry.c:.0f} mm\n", 'normal')
        self.insert_text(f"   • Portées                                  L_x     = {calc.geometry.L_x:.0f} mm, L_y = {calc.geometry.L_y:.0f} mm\n\n", 'normal')
        
        d_champ = calc.geometry.get_d_moyen(True)
        d_dalle = calc.geometry.get_d_moyen(False)
        self.insert_text("   • Hauteur statique (champignon):           d_champ = h_champ - c - φ/2\n", 'normal')
        self.insert_text(f"                                              d_champ = {calc.geometry.h_champignon:.0f} - {calc.geometry.c:.0f} - 25 = {d_champ:.1f} mm\n", 'normal')
        self.insert_text("   • Hauteur statique (dalle courante):       d_dalle = h - c - φ/2\n", 'normal')
        self.insert_text(f"                                              d_dalle = {calc.geometry.h:.0f} - {calc.geometry.c:.0f} - 25 = {d_dalle:.1f} mm\n\n", 'normal')
        
        r_b = calc.geometry.get_rayon_dalle_equivalent()
        self.insert_text("   • Rayon équivalent de dalle:               r_b     = 0.6 × min(L_x, L_y)\n", 'normal')
        self.insert_text(f"                                              r_b     = 0.6 × min({calc.geometry.L_x:.0f}, {calc.geometry.L_y:.0f})\n", 'normal')
        self.insert_text(f"                                              r_b     = {r_b:.1f} mm\n\n", 'normal')
        
        self.insert_text("1.4 ARMATURES\n", 'subheader')
        self.insert_text(f"   • Nappes supérieures:                      ρ_sup   = {calc.reinforcement.rho_sup:.2f} %\n", 'normal')
        self.insert_text(f"                                              Ø_sup   = {calc.reinforcement.phi_sup:.0f} mm\n", 'normal')
        
        esp_sup = calc.reinforcement.get_espacement(calc.reinforcement.phi_sup, calc.reinforcement.rho_sup, d_champ)
        self.insert_text(f"     - Espacement calculé:                    s_sup   = {esp_sup:.0f} mm\n", 'normal')
        
        As_sup = calc.reinforcement.rho_sup * d_champ * 10
        self.insert_text(f"     - Section par mètre:                     A_s,sup = ρ × d × 1000\n", 'normal')
        self.insert_text(f"                                              A_s,sup = {calc.reinforcement.rho_sup:.2f}% × {d_champ:.1f} × 10 = {As_sup:.0f} mm²/m\n\n", 'normal')
        
        self.insert_text(f"   • Nappes inférieures:                      ρ_inf   = {calc.reinforcement.rho_inf:.2f} %\n", 'normal')
        self.insert_text(f"                                              Ø_inf   = {calc.reinforcement.phi_inf:.0f} mm\n", 'normal')
        
        esp_inf = calc.reinforcement.get_espacement(calc.reinforcement.phi_inf, calc.reinforcement.rho_inf, d_dalle)
        self.insert_text(f"     - Espacement calculé:                    s_inf   = {esp_inf:.0f} mm\n", 'normal')
        
        As_inf = calc.reinforcement.rho_inf * d_dalle * 10
        self.insert_text(f"     - Section par mètre:                     A_s,inf = ρ × d × 1000\n", 'normal')
        self.insert_text(f"                                              A_s,inf = {calc.reinforcement.rho_inf:.2f}% × {d_dalle:.1f} × 10 = {As_inf:.0f} mm²/m\n\n", 'normal')
        
        self.insert_text("1.5 SOLLICITATIONS\n", 'subheader')
        self.insert_text(f"   • Effort tranchant de calcul               V_Ed    = {self.var_V_Ed.get():.1f} kN\n\n", 'normal')
        
        # 2. RÉSISTANCES DE CALCUL
        self.insert_text("2. RÉSISTANCES DE CALCUL DU BÉTON\n", 'header')
        self.insert_text("─"*100 + "\n\n", 'normal')
        
        self.insert_text("2.1 RÉSISTANCE DE CALCUL EN COMPRESSION\n", 'subheader')
        eta_fc = min(1.0, 1.0 - calc.materials.f_ck / 250.0)
        self.insert_text(f"   • Coefficient d'efficacité (formule):      η_fc    = min(1.0, 1.0 - f_ck/250)\n", 'normal')
        self.insert_text(f"                                              η_fc    = min(1.0, 1.0 - {calc.materials.f_ck:.1f}/250)\n", 'normal')
        self.insert_text(f"                                              η_fc    = {eta_fc:.3f}\n\n", 'normal')
        
        self.insert_text(f"   • Coefficient de sécurité:                 γ_c     = {calc.gamma_c:.2f}\n", 'normal')
        self.insert_text(f"   • Résistance de calcul (formule):          f_cd    = η_fc × f_ck / γ_c\n", 'normal')
        self.insert_text(f"                                              f_cd    = {eta_fc:.3f} × {calc.materials.f_ck:.1f} / {calc.gamma_c:.2f}\n", 'normal')
        self.insert_text(f"                                              f_cd    = {calc.f_cd:.2f} MPa\n\n", 'normal')
        
        self.insert_text("2.2 CONTRAINTE LIMITE DE CISAILLEMENT\n", 'subheader')
        self.insert_text(f"   • Formule de Muttoni:                      τ_cd    = 0.2 × f_ck\n", 'normal')
        self.insert_text(f"                                              τ_cd    = 0.2 × {calc.materials.f_ck:.1f}\n", 'normal')
        self.insert_text(f"                                              τ_cd    = {calc.tau_cd:.2f} MPa\n\n", 'normal')
        
        # 3. PÉRIMÈTRES DE CONTRÔLE
        self.insert_text("3. PÉRIMÈTRES DE CONTRÔLE\n", 'header')
        self.insert_text("─"*100 + "\n\n", 'normal')
        
        self.insert_text("   Le périmètre de contrôle est situé à une distance d/2 du bord de la colonne.\n\n", 'normal')
        
        self.insert_text("3.1 AU NIVEAU DU CHAMPIGNON\n", 'subheader')
        u_champ = calc.get_perimetre_controle(True)
        
        if calc.geometry.column_type == 'circular':
            self.insert_text(f"   • Formule (colonne circulaire):            u       = π × (d_col + d_champ)\n", 'normal')
            self.insert_text(f"                                              u       = π × ({calc.geometry.d_colonne:.0f} + {d_champ:.1f})\n", 'normal')
        elif calc.geometry.column_type == 'square':
            self.insert_text(f"   • Formule (colonne carrée):                u       = 4 × b + π × d_champ\n", 'normal')
            self.insert_text(f"                                              u       = 4 × {calc.geometry.b_colonne:.0f} + π × {d_champ:.1f}\n", 'normal')
        else:
            self.insert_text(f"   • Formule (colonne rectangulaire):         u       = 2(b + h) + π × d_champ\n", 'normal')
            self.insert_text(f"                                              u       = 2({calc.geometry.b_colonne:.0f} + {calc.geometry.h_colonne:.0f}) + π × {d_champ:.1f}\n", 'normal')
        
        self.insert_text(f"                                              u_champ = {u_champ:.0f} mm\n\n", 'normal')
        
        self.insert_text("3.2 AU NIVEAU DE LA DALLE COURANTE\n", 'subheader')
        u_dalle = calc.get_perimetre_controle(False)
        
        if calc.geometry.column_type == 'circular':
            self.insert_text(f"   • Formule (colonne circulaire):            u       = π × (d_col + d_dalle)\n", 'normal')
            self.insert_text(f"                                              u       = π × ({calc.geometry.d_colonne:.0f} + {d_dalle:.1f})\n", 'normal')
        elif calc.geometry.column_type == 'square':
            self.insert_text(f"   • Formule (colonne carrée):                u       = 4 × b + π × d_dalle\n", 'normal')
            self.insert_text(f"                                              u       = 4 × {calc.geometry.b_colonne:.0f} + π × {d_dalle:.1f}\n", 'normal')
        else:
            self.insert_text(f"   • Formule (colonne rectangulaire):         u       = 2(b + h) + π × d_dalle\n", 'normal')
            self.insert_text(f"                                              u       = 2({calc.geometry.b_colonne:.0f} + {calc.geometry.h_colonne:.0f}) + π × {d_dalle:.1f}\n", 'normal')
        
        self.insert_text(f"                                              u_dalle = {u_dalle:.0f} mm\n\n", 'normal')
        
        # 4. RELATION MOMENT-COURBURE
        self.insert_text("4. RELATION MOMENT-COURBURE\n", 'header')
        self.insert_text("─"*100 + "\n\n", 'normal')
        
        self.insert_text("4.1 MOMENT DE FISSURATION\n", 'subheader')
        m_r = calc.materials.f_ct * (calc.geometry.h**2 / 6) / 1e6
        self.insert_text(f"   • Formule:                                 m_r     = f_ct × h² / 6\n", 'normal')
        self.insert_text(f"                                              m_r     = {calc.materials.f_ct:.2f} × {calc.geometry.h:.0f}² / 6 / 1000000\n", 'normal')
        self.insert_text(f"                                              m_r     = {m_r:.2f} kNm/m\n\n", 'normal')
        
        kappa_r = calc.materials.f_ct / (calc.materials.E_c * d_champ/2) * 1000
        self.insert_text(f"   • Courbure de fissuration:                 κ_r     = f_ct / (E_c × d/2) × 1000\n", 'normal')
        self.insert_text(f"                                              κ_r     = {calc.materials.f_ct:.2f} / ({calc.materials.E_c:.0f} × {d_champ:.1f}/2) × 1000\n", 'normal')
        self.insert_text(f"                                              κ_r     = {kappa_r:.3f} 1/m\n\n", 'normal')
        
        self.insert_text("4.2 MOMENT PLASTIQUE\n", 'subheader')
        rho_dec = calc.reinforcement.get_rho_sup_decimal()
        m_pl = rho_dec * d_champ * calc.f_yd * (d_champ - 0.4 * rho_dec * d_champ * calc.f_yd / calc.f_cd) / 1e3
        
        self.insert_text(f"   • Formule:                                 m_pl    = ρ × d × f_yd × (d - 0.4 × ρ × d × f_yd / f_cd)\n", 'normal')
        self.insert_text(f"                                              m_pl    = {rho_dec:.4f} × {d_champ:.1f} × {calc.f_yd:.1f} × ({d_champ:.1f} - 0.4 × {rho_dec:.4f} × {d_champ:.1f} × {calc.f_yd:.1f} / {calc.f_cd:.2f}) / 1000\n", 'normal')
        self.insert_text(f"                                              m_pl    = {m_pl:.2f} kNm/m\n\n", 'normal')
        
        EI_fissure = calc.materials.E_s * rho_dec * d_champ**3 * (1 - 0.4 * rho_dec) / 1e9
        self.insert_text(f"   • Rigidité après fissuration:              EI_fis  = E_s × ρ × d³ × (1 - 0.4 × ρ) / 10⁹\n", 'normal')
        self.insert_text(f"                                              EI_fis  = {calc.materials.E_s:.0f} × {rho_dec:.4f} × {d_champ:.1f}³ × (1 - 0.4 × {rho_dec:.4f}) / 10⁹\n", 'normal')
        self.insert_text(f"                                              EI_fis  = {EI_fissure:.2f} kNm²/m\n\n", 'normal')
        
        # 5. RÉSISTANCE AU POINÇONNEMENT
        self.insert_text("5. CALCUL DE LA RÉSISTANCE AU POINÇONNEMENT\n", 'header')
        self.insert_text("─"*100 + "\n\n", 'normal')
        
        self.insert_text("   Selon la méthode de Muttoni, la résistance au poinçonnement est déterminée par\n", 'normal')
        self.insert_text("   l'intersection entre:\n", 'normal')
        self.insert_text("   - La courbe force-rotation issue de la relation moment-courbure\n", 'normal')
        self.insert_text("   - Le critère de rupture de Muttoni\n\n", 'normal')
        
        self.insert_text("5.1 CRITÈRE DE RUPTURE DE MUTTONI\n", 'subheader')
        self.insert_text(f"   • Formule générale:                        τ_Rd    = τ_cd × 0.45 / (1.5 × d × ψ × k_Dmax + 0.135)\n\n", 'normal')
        self.insert_text(f"   Avec:  τ_cd = {calc.tau_cd:.2f} MPa\n", 'normal')
        self.insert_text(f"          k_Dmax = {k_Dmax:.3f}\n", 'normal')
        self.insert_text(f"          d en mètres\n", 'normal')
        self.insert_text(f"          ψ = rotation de la dalle en radians\n\n", 'normal')
        
        self.insert_text("5.2 TRANSFORMATION MOMENT-COURBURE EN FORCE-ROTATION\n", 'subheader')
        log_ratio = math.log(max(r_b/r_eq, 2.0))
        factor = 2 * math.pi / log_ratio
        
        self.insert_text(f"   • Facteur de transformation:               α       = 2π / ln(r_b / r_a)\n", 'normal')
        self.insert_text(f"                                              α       = 2π / ln({r_b:.1f} / {r_eq:.1f})\n", 'normal')
        self.insert_text(f"                                              α       = 2π / {log_ratio:.3f} = {factor:.2f}\n\n", 'normal')
        
        self.insert_text(f"   • Force:                                   V       = α × m\n", 'normal')
        self.insert_text(f"   • Rotation:                                ψ       = κ × (r_b - r_a) / 2\n\n", 'normal')
        
        # 6. RÉSULTATS AU NIVEAU DU CHAMPIGNON
        self.insert_text("6. VÉRIFICATION AU NIVEAU DU CHAMPIGNON\n", 'header')
        self.insert_text("─"*100 + "\n\n", 'normal')
        
        tau_Rd_champ = calc.calculate_tau_Rd(psi_Rd_champ, True)
        self.insert_text(f"   • Rotation à la rupture:                   ψ_Rd    = {psi_Rd_champ*1000:.3f} ‰ = {psi_Rd_champ:.6f} rad\n\n", 'normal')
        
        self.insert_text(f"   • Calcul de la contrainte de rupture:\n", 'bold')
        self.insert_text(f"     τ_Rd = τ_cd × 0.45 / (1.5 × d × ψ × k_Dmax + 0.135)\n", 'normal')
        self.insert_text(f"     τ_Rd = {calc.tau_cd:.2f} × 0.45 / (1.5 × {d_champ/1000:.4f} × {psi_Rd_champ:.6f} × {k_Dmax:.3f} + 0.135)\n", 'normal')
        
        denominator_champ = 1.5 * (d_champ/1000) * psi_Rd_champ * k_Dmax + 0.135
        self.insert_text(f"     τ_Rd = {calc.tau_cd * 0.45:.2f} / {denominator_champ:.4f}\n", 'normal')
        self.insert_text(f"     τ_Rd = {tau_Rd_champ:.2f} MPa\n\n", 'normal')
        
        self.insert_text(f"   • Ratio de résistance:                     τ_Rd/τ_cd = {tau_Rd_champ:.2f}/{calc.tau_cd:.2f} = {ratio_champ:.3f}\n\n", 'normal')
        
        self.insert_text(f"   • Calcul de la résistance au poinçonnement:\n", 'bold')
        self.insert_text(f"     V_Rd = τ_Rd × u × d\n", 'normal')
        self.insert_text(f"     V_Rd = {tau_Rd_champ:.2f} × {u_champ:.0f} × {d_champ:.1f} / 1000\n", 'normal')
        self.insert_text(f"     V_Rd = {V_Rd_champ:.1f} kN\n\n", 'normal')
        
        ratio_champ_use = self.var_V_Ed.get() / V_Rd_champ
        self.insert_text(f"   • Vérification:                            V_Ed / V_Rd = {self.var_V_Ed.get():.1f} / {V_Rd_champ:.1f} = {ratio_champ_use:.3f}\n\n", 'normal')
        
        if self.var_V_Ed.get() <= V_Rd_champ:
            self.insert_text("   ✓ VÉRIFICATION OK AU NIVEAU DU CHAMPIGNON\n", 'success')
            self.insert_text(f"   Taux de travail = {ratio_champ_use * 100:.1f} %\n", 'success')
            self.insert_text(f"   Marge de sécurité = {(1 - ratio_champ_use) * 100:.1f} %\n\n", 'success')
        else:
            self.insert_text("   ✗ VÉRIFICATION NON SATISFAITE AU NIVEAU DU CHAMPIGNON\n", 'danger')
            self.insert_text(f"   Dépassement = {(ratio_champ_use - 1) * 100:.1f} %\n", 'danger')
            self.insert_text("   RENFORCEMENT NÉCESSAIRE !\n\n", 'danger')
        
        # 7. RÉSULTATS AU NIVEAU DE LA DALLE COURANTE
        self.insert_text("7. VÉRIFICATION AU NIVEAU DE LA DALLE COURANTE\n", 'header')
        self.insert_text("─"*100 + "\n\n", 'normal')
        
        tau_Rd_dalle = calc.calculate_tau_Rd(psi_Rd_dalle, False)
        self.insert_text(f"   • Rotation à la rupture:                   ψ_Rd    = {psi_Rd_dalle*1000:.3f} ‰ = {psi_Rd_dalle:.6f} rad\n\n", 'normal')
        
        self.insert_text(f"   • Calcul de la contrainte de rupture:\n", 'bold')
        self.insert_text(f"     τ_Rd = τ_cd × 0.45 / (1.5 × d × ψ × k_Dmax + 0.135)\n", 'normal')
        self.insert_text(f"     τ_Rd = {calc.tau_cd:.2f} × 0.45 / (1.5 × {d_dalle/1000:.4f} × {psi_Rd_dalle:.6f} × {k_Dmax:.3f} + 0.135)\n", 'normal')
        
        denominator_dalle = 1.5 * (d_dalle/1000) * psi_Rd_dalle * k_Dmax + 0.135
        self.insert_text(f"     τ_Rd = {calc.tau_cd * 0.45:.2f} / {denominator_dalle:.4f}\n", 'normal')
        self.insert_text(f"     τ_Rd = {tau_Rd_dalle:.2f} MPa\n\n", 'normal')
        
        self.insert_text(f"   • Ratio de résistance:                     τ_Rd/τ_cd = {tau_Rd_dalle:.2f}/{calc.tau_cd:.2f} = {ratio_dalle:.3f}\n\n", 'normal')
        
        self.insert_text(f"   • Calcul de la résistance au poinçonnement:\n", 'bold')
        self.insert_text(f"     V_Rd = τ_Rd × u × d\n", 'normal')
        self.insert_text(f"     V_Rd = {tau_Rd_dalle:.2f} × {u_dalle:.0f} × {d_dalle:.1f} / 1000\n", 'normal')
        self.insert_text(f"     V_Rd = {V_Rd_dalle:.1f} kN\n\n", 'normal')
        
        ratio_dalle_use = self.var_V_Ed.get() / V_Rd_dalle
        self.insert_text(f"   • Vérification:                            V_Ed / V_Rd = {self.var_V_Ed.get():.1f} / {V_Rd_dalle:.1f} = {ratio_dalle_use:.3f}\n\n", 'normal')
        
        if self.var_V_Ed.get() <= V_Rd_dalle:
            self.insert_text("   ✓ VÉRIFICATION OK AU NIVEAU DE LA DALLE COURANTE\n", 'success')
            self.insert_text(f"   Taux de travail = {ratio_dalle_use * 100:.1f} %\n", 'success')
            self.insert_text(f"   Marge de sécurité = {(1 - ratio_dalle_use) * 100:.1f} %\n\n", 'success')
        else:
            self.insert_text("   ✗ VÉRIFICATION NON SATISFAITE AU NIVEAU DE LA DALLE COURANTE\n", 'danger')
            self.insert_text(f"   Dépassement = {(ratio_dalle_use - 1) * 100:.1f} %\n", 'danger')
            self.insert_text("   RENFORCEMENT NÉCESSAIRE !\n\n", 'danger')
        
        # 8. VÉRIFICATION GLOBALE
        self.insert_text("8. VÉRIFICATION GLOBALE\n", 'header')
        self.insert_text("─"*100 + "\n\n", 'normal')
        
        V_Rd_min = min(V_Rd_champ, V_Rd_dalle)
        zone_critique = 'CHAMPIGNON' if V_Rd_champ < V_Rd_dalle else 'DALLE COURANTE'
        
        self.insert_text(f"   • Résistance au champignon:                V_Rd,champ = {V_Rd_champ:.1f} kN\n", 'normal')
        self.insert_text(f"   • Résistance à la dalle courante:          V_Rd,dalle = {V_Rd_dalle:.1f} kN\n\n", 'normal')
        
        self.insert_text(f"   • Résistance minimale retenue:             V_Rd,min   = min({V_Rd_champ:.1f}, {V_Rd_dalle:.1f})\n", 'normal')
        self.insert_text(f"                                              V_Rd,min   = {V_Rd_min:.1f} kN\n\n", 'normal')
        
        if zone_critique == 'CHAMPIGNON':
            self.insert_text(f"   • Zone critique:                           {zone_critique}\n", 'normal')
        else:
            self.insert_text(f"   • Zone critique:                           {zone_critique}\n", 'warning')
            self.insert_text("     (Attention: la dalle courante est plus critique que le champignon)\n", 'warning')
        
        self.insert_text("\n   • VÉRIFICATION FINALE:\n", 'bold')
        self.insert_text(f"     Condition:  V_Ed ≤ V_Rd,min\n", 'normal')
        self.insert_text(f"                 {self.var_V_Ed.get():.1f} kN ≤ {V_Rd_min:.1f} kN\n\n", 'normal')
        
        if self.var_V_Ed.get() <= V_Rd_min:
            self.insert_text("   " + "="*80 + "\n", 'success')
            self.insert_text("   ✓✓✓ DALLE VÉRIFIÉE AU POINÇONNEMENT ✓✓✓\n", 'success')
            self.insert_text("   " + "="*80 + "\n\n", 'success')
        else:
            self.insert_text("   " + "="*80 + "\n", 'danger')
            self.insert_text("   ✗✗✗ DALLE NON VÉRIFIÉE - RENFORCEMENT REQUIS ✗✗✗\n", 'danger')
            self.insert_text("   " + "="*80 + "\n\n", 'danger')
        
        # 9. VÉRIFICATIONS COMPLÉMENTAIRES
        self.insert_text("9. VÉRIFICATIONS COMPLÉMENTAIRES\n", 'header')
        self.insert_text("─"*100 + "\n\n", 'normal')
        
        self.insert_text("9.1 ARMATURE MINIMALE\n", 'subheader')
        rho_min = max(0.26 * calc.materials.f_ct / calc.materials.f_yk, 0.0013) * 100
        
        self.insert_text(f"   • Formule:                                 ρ_min   = max(0.26 × f_ct / f_yk, 0.0013) × 100\n", 'normal')
        self.insert_text(f"                                              ρ_min   = max(0.26 × {calc.materials.f_ct:.2f} / {calc.materials.f_yk:.0f}, 0.0013) × 100\n", 'normal')
        self.insert_text(f"                                              ρ_min   = {rho_min:.3f} %\n\n", 'normal')
        
        self.insert_text(f"   • Armature supérieure:                     ρ_sup   = {calc.reinforcement.rho_sup:.3f} %\n", 'normal')
        if calc.reinforcement.rho_sup >= rho_min:
            self.insert_text(f"     ✓ OK (ρ_sup ≥ ρ_min)\n\n", 'success')
        else:
            self.insert_text(f"     ✗ INSUFFISANT (ρ_sup < ρ_min)\n", 'danger')
            self.insert_text(f"     Augmenter à au moins {rho_min:.3f} %\n\n", 'danger')
        
        self.insert_text(f"   • Armature inférieure:                     ρ_inf   = {calc.reinforcement.rho_inf:.3f} %\n", 'normal')
        if calc.reinforcement.rho_inf >= rho_min:
            self.insert_text(f"     ✓ OK (ρ_inf ≥ ρ_min)\n\n", 'success')
        else:
            self.insert_text(f"     ✗ INSUFFISANT (ρ_inf < ρ_min)\n", 'danger')
            self.insert_text(f"     Augmenter à au moins {rho_min:.3f} %\n\n", 'danger')
        
        self.insert_text("9.2 DUCTILITÉ\n", 'subheader')
        self.insert_text(f"   • Rotation critique:                       ψ_critique = 5 ‰\n", 'normal')
        self.insert_text(f"   • Rotation de rupture (champignon):        ψ_Rd,champ = {psi_Rd_champ*1000:.3f} ‰\n", 'normal')
        
        if psi_Rd_champ >= 0.005:
            self.insert_text(f"     ✓ Comportement ductile (ψ_Rd ≥ 5 ‰)\n\n", 'success')
        else:
            self.insert_text(f"     ⚠ ATTENTION: Comportement fragile possible (ψ_Rd < 5 ‰)\n", 'warning')
            self.insert_text(f"     Envisager d'augmenter le taux d'armature pour améliorer la ductilité\n\n", 'warning')
        
        self.insert_text("9.3 ESPACEMENTS DES ARMATURES\n", 'subheader')
        self.insert_text(f"   • Espacement maximal recommandé:           s_max   = 2 × h ≤ 300 mm\n", 'normal')
        s_max = min(2 * calc.geometry.h_champignon, 300)
        self.insert_text(f"                                              s_max   = min(2 × {calc.geometry.h_champignon:.0f}, 300) = {s_max:.0f} mm\n\n", 'normal')
        
        self.insert_text(f"   • Espacement supérieur:                    s_sup   = {esp_sup:.0f} mm\n", 'normal')
        if esp_sup <= s_max:
            self.insert_text(f"     ✓ OK (s_sup ≤ s_max)\n\n", 'success')
        else:
            self.insert_text(f"     ✗ Trop grand (s_sup > s_max)\n", 'warning')
            self.insert_text(f"     Réduire à maximum {s_max:.0f} mm\n\n", 'warning')
        
        self.insert_text(f"   • Espacement inférieur:                    s_inf   = {esp_inf:.0f} mm\n", 'normal')
        if esp_inf <= s_max:
            self.insert_text(f"     ✓ OK (s_inf ≤ s_max)\n\n", 'success')
        else:
            self.insert_text(f"     ✗ Trop grand (s_inf > s_max)\n", 'warning')
            self.insert_text(f"     Réduire à maximum {s_max:.0f} mm\n\n", 'warning')
        
        # 10. RECOMMANDATIONS
        self.insert_text("10. RECOMMANDATIONS ET REMARQUES\n", 'header')
        self.insert_text("─"*100 + "\n\n", 'normal')
        
        recommendations = []
        
        if V_Rd_min < self.var_V_Ed.get():
            recommendations.append("• PRIORITAIRE: Renforcement structurel nécessaire (résistance insuffisante)")
            recommendations.append("  Solutions possibles:")
            recommendations.append("    - Augmenter l'épaisseur du champignon")
            recommendations.append("    - Augmenter le rayon du champignon")
            recommendations.append("    - Augmenter le taux d'armature")
            recommendations.append("    - Utiliser un béton de classe supérieure")
            recommendations.append("    - Ajouter des armatures de poinçonnement (étriers, goujons)")
        
        if zone_critique == 'DALLE COURANTE':
            recommendations.append("• La dalle courante est plus critique que le champignon")
            recommendations.append("  Considérer l'augmentation de l'épaisseur de la dalle courante")
        
        if psi_Rd_champ < 0.005:
            recommendations.append("• Ductilité insuffisante: augmenter le taux d'armature supérieur")
        
        if calc.reinforcement.rho_sup < rho_min:
            recommendations.append(f"• Armature supérieure inférieure au minimum: porter à {rho_min:.3f} % minimum")
        
        if calc.reinforcement.rho_inf < rho_min:
            recommendations.append(f"• Armature inférieure inférieure au minimum: porter à {rho_min:.3f} % minimum")
        
        if esp_sup > s_max:
            recommendations.append(f"• Espacement armature supérieure trop grand: réduire à {s_max:.0f} mm maximum")
        
        if esp_inf > s_max:
            recommendations.append(f"• Espacement armature inférieure trop grand: réduire à {s_max:.0f} mm maximum")
        
        if not recommendations:
            self.insert_text("   ✓ Aucune remarque particulière\n", 'success')
            self.insert_text("   La dalle respecte toutes les vérifications\n\n", 'success')
        else:
            for rec in recommendations:
                if "PRIORITAIRE" in rec:
                    self.insert_text(f"   {rec}\n", 'danger')
                elif "⚠" in rec or "ATTENTION" in rec:
                    self.insert_text(f"   {rec}\n", 'warning')
                else:
                    self.insert_text(f"   {rec}\n", 'normal')
            self.insert_text("\n", 'normal')
        
        # Footer
        self.insert_text("="*100 + "\n", 'header')
        self.insert_text("                          FIN DU RAPPORT DE CALCUL\n", 'header')
        self.insert_text("="*100 + "\n", 'header')
        self.insert_text("\n", 'normal')
        self.insert_text("NOTES:\n", 'bold')
        self.insert_text("• Ce rapport a été généré automatiquement selon la méthode de Muttoni\n", 'normal')
        self.insert_text("• Les calculs sont conformes à la thèse EPFL 3380 (2005)\n", 'normal')
        self.insert_text("• Vérifier les hypothèses et résultats avant utilisation\n", 'normal')
        self.insert_text("• Consulter un ingénieur structure pour validation finale\n", 'normal')
    
    def insert_text(self, text, tag):
        """Insère du texte avec le tag approprié"""
        self.text_results.insert(tk.END, text, tag)
    
    def update_graphs(self):
        """Met à jour les graphiques"""
        if not hasattr(self, 'calculator'):
            return
        
        for widget in self.frame_plot.winfo_children():
            widget.destroy()
        
        fig = Figure(figsize=(16, 10))
        fig.patch.set_facecolor(COLORS['white'])
        
        # Graphique 1: Critère de rupture (champignon)
        ax1 = fig.add_subplot(2, 2, 1)
        self.plot_failure_criterion(ax1, at_champignon=True)
        
        # Graphique 2: Critère de rupture (dalle)
        ax2 = fig.add_subplot(2, 2, 2)
        self.plot_failure_criterion(ax2, at_champignon=False)
        
        # Graphique 3: Moment-courbure
        ax3 = fig.add_subplot(2, 2, 3)
        self.plot_moment_curvature(ax3)
        
        # Graphique 4: Comparaison
        ax4 = fig.add_subplot(2, 2, 4)
        self.plot_resistance_comparison(ax4)
        
        fig.tight_layout(pad=4.0, rect=[0, 0, 1, 0.98])
        
        canvas = FigureCanvasTkAgg(fig, master=self.frame_plot)
        canvas.draw()
        canvas.get_tk_widget().pack(fill='both', expand=True)
        
        self.current_figure = fig
    
    def plot_failure_criterion(self, ax, at_champignon: bool):
        """Trace le critère de rupture"""
        calc = self.calculator
        
        psi_values, V_curve = calc.calculate_load_rotation_curve(at_champignon)
        
        psi_criterion = np.linspace(0.001, 0.05, 100)
        V_criterion = [calc.calculate_V_Rd(psi, at_champignon) for psi in psi_criterion]
        V_criterion = np.array(V_criterion)
        
        V_Rd, psi_Rd, _ = calc.find_punching_resistance(at_champignon)
        
        ax.plot(psi_criterion * 1000, V_criterion, 'r-', linewidth=2.5, label='Critère Muttoni')
        ax.plot(psi_values * 1000, V_curve, color=COLORS['secondary'], linewidth=2.5, label='Comportement flexion')
        ax.plot(psi_Rd * 1000, V_Rd, 'o', color=COLORS['success'], markersize=12, label=f'Rupture: {V_Rd:.1f} kN')
        
        V_Ed = self.var_V_Ed.get()
        ax.axhline(y=V_Ed, color=COLORS['warning'], linestyle='--', linewidth=2, label=f'V_Ed = {V_Ed:.1f} kN')
        
        ax.set_xlabel('Rotation ψ [‰]', fontsize=11, fontweight='bold')
        ax.set_ylabel('Force V [kN]', fontsize=11, fontweight='bold')
        title = 'CHAMPIGNON' if at_champignon else 'DALLE COURANTE'
        ax.set_title(f'Diagramme Force-Rotation\n{title}', fontsize=11, fontweight='bold', pad=15)
        ax.legend(fontsize=9, loc='best')
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.set_xlim(0, max(psi_criterion) * 1000)
        ax.set_ylim(0, max(max(V_criterion), max(V_curve)) * 1.2)
        
        ax.set_facecolor('#F8F9FA')
    
    def plot_moment_curvature(self, ax):
        """Trace moment-courbure"""
        calc = self.calculator
        
        kappa_champ, m_champ = calc.calculate_moment_curvature_relation(at_champignon=True)
        kappa_dalle, m_dalle = calc.calculate_moment_curvature_relation(at_champignon=False)
        
        ax.plot(kappa_champ, m_champ, color=COLORS['secondary'], linewidth=2.5, label='Champignon')
        ax.plot(kappa_dalle, m_dalle, color=COLORS['danger'], linewidth=2.5, linestyle='--', label='Dalle courante')
        
        ax.set_xlabel('Courbure κ [1/m]', fontsize=11, fontweight='bold')
        ax.set_ylabel('Moment m [kNm/m]', fontsize=11, fontweight='bold')
        ax.set_title('Relation Moment-Courbure', fontsize=11, fontweight='bold', pad=15)
        ax.legend(fontsize=10, loc='best')
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.set_facecolor('#F8F9FA')
    
    def plot_resistance_comparison(self, ax):
        """Graphique comparatif"""
        calc = self.calculator
        
        V_Rd_champ, _, _ = calc.find_punching_resistance(at_champignon=True)
        V_Rd_dalle, _, _ = calc.find_punching_resistance(at_champignon=False)
        V_Ed = self.var_V_Ed.get()
        
        categories = ['V_Ed\n(Sollicitation)', 'V_Rd\n(Champignon)', 'V_Rd\n(Dalle)']
        values = [V_Ed, V_Rd_champ, V_Rd_dalle]
        colors = [COLORS['warning'], 
                 COLORS['success'] if V_Ed <= V_Rd_champ else COLORS['danger'],
                 COLORS['success'] if V_Ed <= V_Rd_dalle else COLORS['danger']]
        
        bars = ax.bar(categories, values, color=colors, alpha=0.8, edgecolor='black', linewidth=2)
        
        for bar, value in zip(bars, values):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{value:.1f} kN',
                   ha='center', va='bottom', fontweight='bold', fontsize=11)
        
        ax.set_ylabel('Force [kN]', fontsize=11, fontweight='bold')
        ax.set_title('Comparaison des Résistances', fontsize=11, fontweight='bold', pad=15)
        ax.grid(True, axis='y', alpha=0.3, linestyle='--')
        ax.axhline(y=V_Ed, color=COLORS['warning'], linestyle='--', linewidth=2)
        ax.set_facecolor('#F8F9FA')
    
    def draw_reinforcement_plan(self):
        """Dessine les plans d'armature"""
        if not hasattr(self, 'calculator'):
            return
        
        for widget in self.frame_reinf_canvas.winfo_children():
            widget.destroy()
        
        fig = Figure(figsize=(16, 10))
        fig.patch.set_facecolor(COLORS['white'])
        
        ax1 = fig.add_subplot(2, 2, 1)
        self.draw_plan_view(ax1)
        
        ax2 = fig.add_subplot(2, 2, 2)
        self.draw_section_view(ax2)
        
        ax3 = fig.add_subplot(2, 2, 3)
        self.draw_reinforcement_detail(ax3, nappe='supérieure')
        
        ax4 = fig.add_subplot(2, 2, 4)
        self.draw_reinforcement_detail(ax4, nappe='inférieure')
        
        fig.tight_layout(pad=4.0, rect=[0, 0, 1, 0.98])
        
        canvas = FigureCanvasTkAgg(fig, master=self.frame_reinf_canvas)
        canvas.draw()
        canvas.get_tk_widget().pack(fill='both', expand=True)
        
        self.current_reinforcement_figure = fig
    
    def draw_plan_view(self, ax):
        """Vue en plan avec type de colonne"""
        geom = self.calculator.geometry
        
        dalle_size = min(geom.L_x, geom.L_y) / 2
        dalle_rect = plt.Rectangle((-dalle_size/2, -dalle_size/2), dalle_size, dalle_size,
                                   fill=True, facecolor='lightgray', edgecolor='black', linewidth=2)
        ax.add_patch(dalle_rect)
        
        champignon_circle = plt.Circle((0, 0), geom.r_champignon, 
                                       fill=True, facecolor='wheat', edgecolor='brown', linewidth=2)
        ax.add_patch(champignon_circle)
        
        if geom.column_type == 'circular':
            col_circle = plt.Circle((0, 0), geom.d_colonne/2,
                                   fill=True, facecolor='darkgray', edgecolor='black', linewidth=2)
            ax.add_patch(col_circle)
        elif geom.column_type == 'square':
            col_rect = plt.Rectangle((-geom.b_colonne/2, -geom.b_colonne/2), 
                                    geom.b_colonne, geom.b_colonne,
                                    fill=True, facecolor='darkgray', edgecolor='black', linewidth=2)
            ax.add_patch(col_rect)
        else:
            col_rect = plt.Rectangle((-geom.b_colonne/2, -geom.h_colonne/2), 
                                    geom.b_colonne, geom.h_colonne,
                                    fill=True, facecolor='darkgray', edgecolor='black', linewidth=2)
            ax.add_patch(col_rect)
        
        d_champ = geom.get_d_moyen(True)
        if geom.column_type == 'circular':
            perim_circle = plt.Circle((0, 0), geom.d_colonne/2 + d_champ/2,
                                     fill=False, edgecolor='red', linewidth=2, linestyle='--')
            ax.add_patch(perim_circle)
        elif geom.column_type == 'square':
            perim_size = geom.b_colonne + d_champ
            perim_rect = plt.Rectangle((-perim_size/2, -perim_size/2), perim_size, perim_size,
                                      fill=False, edgecolor='red', linewidth=2, linestyle='--',
                                      joinstyle='round')
            ax.add_patch(perim_rect)
        else:
            perim_b = geom.b_colonne + d_champ
            perim_h = geom.h_colonne + d_champ
            perim_rect = plt.Rectangle((-perim_b/2, -perim_h/2), perim_b, perim_h,
                                      fill=False, edgecolor='red', linewidth=2, linestyle='--',
                                      joinstyle='round')
            ax.add_patch(perim_rect)
        
        n_bars = 8
        for i in range(n_bars):
            angle = 2 * math.pi * i / n_bars
            x1 = geom.r_champignon * math.cos(angle) * 1.2
            y1 = geom.r_champignon * math.sin(angle) * 1.2
            x2 = -x1
            y2 = -y1
            ax.plot([x1, x2], [y1, y2], 'b-', linewidth=0.5, alpha=0.5)
        
        if geom.column_type == 'circular':
            ax.annotate('', xy=(geom.d_colonne/2, -dalle_size/2 - 100), 
                       xytext=(-geom.d_colonne/2, -dalle_size/2 - 100),
                       arrowprops=dict(arrowstyle='<->', color='black', lw=1.5))
            ax.text(0, -dalle_size/2 - 150, f'd = {geom.d_colonne:.0f} mm', 
                   ha='center', fontsize=9, fontweight='bold')
        elif geom.column_type == 'square':
            ax.annotate('', xy=(geom.b_colonne/2, -dalle_size/2 - 100), 
                       xytext=(-geom.b_colonne/2, -dalle_size/2 - 100),
                       arrowprops=dict(arrowstyle='<->', color='black', lw=1.5))
            ax.text(0, -dalle_size/2 - 150, f'b = {geom.b_colonne:.0f} mm', 
                   ha='center', fontsize=9, fontweight='bold')
        else:
            ax.annotate('', xy=(geom.b_colonne/2, -dalle_size/2 - 100), 
                       xytext=(-geom.b_colonne/2, -dalle_size/2 - 100),
                       arrowprops=dict(arrowstyle='<->', color='black', lw=1.5))
            ax.text(0, -dalle_size/2 - 150, f'b = {geom.b_colonne:.0f} mm', 
                   ha='center', fontsize=9, fontweight='bold')
        
        ax.annotate('', xy=(0, geom.r_champignon), xytext=(0, 0),
                   arrowprops=dict(arrowstyle='<->', color='brown', lw=1.5))
        ax.text(50, geom.r_champignon/2, f'r = {geom.r_champignon:.0f}', 
               ha='left', fontsize=9, color='brown', fontweight='bold')
        
        ax.set_xlim(-dalle_size/2 - 200, dalle_size/2 + 200)
        ax.set_ylim(-dalle_size/2 - 300, dalle_size/2 + 200)
        ax.set_aspect('equal')
        ax.set_title('VUE EN PLAN\nDalle Champignon', fontsize=10, fontweight='bold', pad=15)
        ax.grid(True, alpha=0.2)
        ax.set_xlabel('x [mm]', fontsize=9)
        ax.set_ylabel('y [mm]', fontsize=9)
        ax.set_facecolor('#F8F9FA')
        
        col_type_names = {'circular': 'Colonne circulaire', 'square': 'Colonne carrée', 
                         'rectangular': 'Colonne rectangulaire'}
        ax.plot([], [], 's', color='darkgray', markersize=8, label=col_type_names[geom.column_type])
        ax.plot([], [], color='brown', linewidth=4, label='Champignon')
        ax.plot([], [], 'r--', linewidth=2, label='Périmètre contrôle')
        ax.legend(loc='upper right', fontsize=8)
    
    def draw_section_view(self, ax):
        """Coupe transversale"""
        geom = self.calculator.geometry
        reinf = self.calculator.reinforcement
        
        L_total = min(geom.L_x, geom.L_y) / 2
        
        ax.fill_between([-L_total, L_total], 0, geom.h, 
                       facecolor='lightgray', edgecolor='black', linewidth=1.5)
        
        x_champ = [-geom.r_champignon, -geom.r_champignon, 
                   -geom.b_colonne/2 if geom.column_type != 'circular' else -geom.d_colonne/2, 
                   geom.b_colonne/2 if geom.column_type != 'circular' else geom.d_colonne/2, 
                   geom.r_champignon, geom.r_champignon]
        y_champ = [geom.h, geom.h_champignon, 
                   geom.h_champignon, geom.h_champignon,
                   geom.h_champignon, geom.h]
        
        ax.fill(x_champ, y_champ, facecolor='wheat', edgecolor='brown', linewidth=2)
        
        if geom.column_type == 'circular':
            ax.fill_between([-geom.d_colonne/2, geom.d_colonne/2], 
                           0, -200, facecolor='darkgray', edgecolor='black', linewidth=2)
        else:
            col_width = geom.b_colonne
            ax.fill_between([-col_width/2, col_width/2], 
                           0, -200, facecolor='darkgray', edgecolor='black', linewidth=2)
        
        y_sup = geom.h_champignon - geom.c - reinf.phi_sup/2
        n_bars_sup = 15
        for i in range(n_bars_sup):
            x = -geom.r_champignon + (2 * geom.r_champignon * i / (n_bars_sup - 1))
            circle = plt.Circle((x, y_sup), reinf.phi_sup/2, 
                              color='red', zorder=10)
            ax.add_patch(circle)
        
        y_inf = geom.c + reinf.phi_inf/2
        n_bars_inf = 12
        for i in range(n_bars_inf):
            x = -L_total/2 + (L_total * i / (n_bars_inf - 1))
            circle = plt.Circle((x, y_inf), reinf.phi_inf/2, 
                              color='blue', zorder=10)
            ax.add_patch(circle)
        
        ax.annotate('', xy=(L_total + 200, 0), xytext=(L_total + 200, geom.h),
                   arrowprops=dict(arrowstyle='<->', color='black', lw=1.5))
        ax.text(L_total + 300, geom.h/2, f'h = {geom.h:.0f}', 
               rotation=90, va='center', fontsize=9, fontweight='bold')
        
        ax.annotate('', xy=(L_total + 200, geom.h), xytext=(L_total + 200, geom.h_champignon),
                   arrowprops=dict(arrowstyle='<->', color='brown', lw=1.5))
        ax.text(L_total + 300, (geom.h + geom.h_champignon)/2, 
               f'{geom.h_champignon:.0f}', rotation=90, va='center', 
               fontsize=9, color='brown', fontweight='bold')
        
        ax.set_xlim(-L_total - 100, L_total + 500)
        ax.set_ylim(-300, geom.h_champignon + 100)
        ax.set_aspect('equal')
        ax.set_title('COUPE A-A\nSection transversale', fontsize=10, fontweight='bold', pad=15)
        ax.grid(True, alpha=0.2)
        ax.set_xlabel('x [mm]', fontsize=9)
        ax.set_ylabel('z [mm]', fontsize=9)
        ax.set_facecolor('#F8F9FA')
        
        ax.plot([], [], 'o', color='red', markersize=8, label=f'Arm. sup. Ø{reinf.phi_sup:.0f}')
        ax.plot([], [], 'o', color='blue', markersize=8, label=f'Arm. inf. Ø{reinf.phi_inf:.0f}')
        ax.legend(loc='upper left', fontsize=8)
    
    def draw_reinforcement_detail(self, ax, nappe: str):
        """Détail des armatures"""
        geom = self.calculator.geometry
        reinf = self.calculator.reinforcement
        
        if nappe == 'supérieure':
            phi = reinf.phi_sup
            rho = reinf.rho_sup
            color = 'red'
            d = geom.get_d_moyen(True)
        else:
            phi = reinf.phi_inf
            rho = reinf.rho_inf
            color = 'blue'
            d = geom.get_d_moyen(False)
        
        espacement = reinf.get_espacement(phi, rho, d)
        
        L_zone = 2000
        n_bars = int(L_zone / espacement) + 1
        
        for i in range(n_bars):
            x = -L_zone/2 + i * espacement
            ax.plot([x, x], [-L_zone/2, L_zone/2], color=color, linewidth=2, alpha=0.6)
            ax.plot([-L_zone/2, L_zone/2], [x, x], color=color, linewidth=2, alpha=0.6)
        
        for i in range(n_bars):
            for j in range(n_bars):
                x = -L_zone/2 + i * espacement
                y = -L_zone/2 + j * espacement
                circle = plt.Circle((x, y), phi/2, color=color, zorder=10)
                ax.add_patch(circle)
        
        if geom.column_type == 'circular':
            col_shape = plt.Circle((0, 0), geom.d_colonne/2,
                                  fill=True, facecolor='lightgray', 
                                  edgecolor='black', linewidth=2, alpha=0.5)
        elif geom.column_type == 'square':
            col_shape = plt.Rectangle((-geom.b_colonne/2, -geom.b_colonne/2), 
                                     geom.b_colonne, geom.b_colonne,
                                     fill=True, facecolor='lightgray', 
                                     edgecolor='black', linewidth=2, alpha=0.5)
        else:
            col_shape = plt.Rectangle((-geom.b_colonne/2, -geom.h_colonne/2), 
                                     geom.b_colonne, geom.h_colonne,
                                     fill=True, facecolor='lightgray', 
                                     edgecolor='black', linewidth=2, alpha=0.5)
        ax.add_patch(col_shape)
        
        y_cot = L_zone/2 + 100
        ax.annotate('', xy=(espacement/2, y_cot), xytext=(-espacement/2, y_cot),
                   arrowprops=dict(arrowstyle='<->', color='black', lw=1.5))
        ax.text(0, y_cot + 50, f's = {espacement:.0f} mm', 
               ha='center', fontsize=10, fontweight='bold')
        
        info_text = f"""NAPPE {nappe.upper()}

Diamètre: Ø{phi:.0f} mm
Taux: ρ = {rho:.2f} %
Espacement: s = {espacement:.0f} mm
Section/m: {rho * d * 10:.0f} mm²/m

Disposition:
- Grille orthogonale
- Recouvrement
- Enrobage: {geom.c:.0f} mm"""
        
        ax.text(L_zone/2 + 200, 0, info_text, 
               fontsize=8, verticalalignment='center',
               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7))
        
        ax.set_xlim(-L_zone/2 - 100, L_zone/2 + 600)
        ax.set_ylim(-L_zone/2 - 200, L_zone/2 + 200)
        ax.set_aspect('equal')
        ax.set_title(f'DÉTAIL ARMATURES\n{nappe.upper()}', fontsize=10, fontweight='bold', pad=15)
        ax.grid(True, alpha=0.2)
        ax.set_xlabel('x [mm]', fontsize=9)
        ax.set_ylabel('y [mm]', fontsize=9)
        ax.set_facecolor('#F8F9FA')
    
    def export_report_pdf(self):
        """Exporte la note de calcul en PDF"""
        if not hasattr(self, 'calculator'):
            messagebox.showwarning("Attention", "Veuillez d'abord effectuer un calcul!")
            return
        
        filename = filedialog.asksaveasfilename(
            defaultextension=".pdf",
            filetypes=[("PDF files", "*.pdf")],
            initialfile=f"Note_Calcul_Poinconnement_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"
        )
        
        if filename:
            try:
                with PdfPages(filename) as pdf:
                    fig = plt.figure(figsize=(8.27, 11.69))
                    fig.text(0.1, 0.95, 'NOTE DE CALCUL - POINÇONNEMENT', 
                            fontsize=16, fontweight='bold')
                    
                    text_content = self.text_results.get(1.0, tk.END)
                    
                    fig.text(0.1, 0.05, text_content, 
                            fontsize=7, family='monospace', verticalalignment='top',
                            wrap=True)
                    
                    pdf.savefig(fig, bbox_inches='tight')
                    plt.close(fig)
                
                messagebox.showinfo("Succès", f"Note de calcul exportée:\n{filename}")
            except Exception as e:
                messagebox.showerror("Erreur", f"Erreur lors de l'export:\n{str(e)}")
    
    def export_graphs_pdf(self):
        """Exporte les graphiques en PDF"""
        if not hasattr(self, 'current_figure'):
            messagebox.showwarning("Attention", "Aucun graphique à exporter!")
            return
        
        filename = filedialog.asksaveasfilename(
            defaultextension=".pdf",
            filetypes=[("PDF files", "*.pdf")],
            initialfile=f"Graphiques_Poinconnement_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"
        )
        
        if filename:
            try:
                self.current_figure.savefig(filename, bbox_inches='tight', dpi=300)
                messagebox.showinfo("Succès", f"Graphiques exportés:\n{filename}")
            except Exception as e:
                messagebox.showerror("Erreur", f"Erreur lors de l'export:\n{str(e)}")
    
    def export_reinforcement_pdf(self):
        """Exporte les plans d'armature en PDF"""
        if not hasattr(self, 'current_reinforcement_figure'):
            messagebox.showwarning("Attention", "Aucun plan à exporter!")
            return
        
        filename = filedialog.asksaveasfilename(
            defaultextension=".pdf",
            filetypes=[("PDF files", "*.pdf")],
            initialfile=f"Plans_Armature_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"
        )
        
        if filename:
            try:
                self.current_reinforcement_figure.savefig(filename, bbox_inches='tight', dpi=300)
                messagebox.showinfo("Succès", f"Plans d'armature exportés:\n{filename}")
            except Exception as e:
                messagebox.showerror("Erreur", f"Erreur lors de l'export:\n{str(e)}")
    
    def export_all_pdf(self):
        """Exporte tout en un seul PDF"""
        if not hasattr(self, 'calculator'):
            messagebox.showwarning("Attention", "Veuillez d'abord effectuer un calcul!")
            return
        
        filename = filedialog.asksaveasfilename(
            defaultextension=".pdf",
            filetypes=[("PDF files", "*.pdf")],
            initialfile=f"Rapport_Complet_Poinconnement_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"
        )
        
        if filename:
            try:
                with PdfPages(filename) as pdf:
                    fig_cover = plt.figure(figsize=(8.27, 11.69))
                    fig_cover.text(0.5, 0.7, 'RAPPORT DE CALCUL', 
                                  ha='center', fontsize=24, fontweight='bold', 
                                  color=COLORS['primary'])
                    fig_cover.text(0.5, 0.65, 'VÉRIFICATION AU POINÇONNEMENT', 
                                  ha='center', fontsize=18, fontweight='bold',
                                  color=COLORS['secondary'])
                    fig_cover.text(0.5, 0.6, 'Dalle Champignon - Méthode Muttoni', 
                                  ha='center', fontsize=14, color=COLORS['dark'])
                    fig_cover.text(0.5, 0.4, f"Date: {datetime.now().strftime('%d/%m/%Y %H:%M')}", 
                                  ha='center', fontsize=12)
                    
                    col_type_names = {'circular': 'Colonne Circulaire', 
                                    'square': 'Colonne Carrée', 
                                    'rectangular': 'Colonne Rectangulaire'}
                    fig_cover.text(0.5, 0.35, f"Type: {col_type_names[self.calculator.geometry.column_type]}", 
                                  ha='center', fontsize=12, color=COLORS['info'])
                    
                    pdf.savefig(fig_cover, bbox_inches='tight')
                    plt.close(fig_cover)
                    
                    fig_text = plt.figure(figsize=(8.27, 11.69))
                    text_content = self.text_results.get(1.0, tk.END)
                    fig_text.text(0.05, 0.95, text_content, 
                                fontsize=6, family='monospace', 
                                verticalalignment='top', wrap=True)
                    pdf.savefig(fig_text, bbox_inches='tight')
                    plt.close(fig_text)
                    
                    if hasattr(self, 'current_figure'):
                        pdf.savefig(self.current_figure, bbox_inches='tight')
                    
                    if hasattr(self, 'current_reinforcement_figure'):
                        pdf.savefig(self.current_reinforcement_figure, bbox_inches='tight')
                    
                    d = pdf.infodict()
                    d['Title'] = 'Rapport de Calcul - Poinçonnement'
                    d['Author'] = 'Logiciel de Calcul Structural'
                    d['Subject'] = 'Vérification au poinçonnement - Méthode Muttoni'
                    d['Keywords'] = 'Poinçonnement, Dalle, Champignon, Béton armé'
                    d['CreationDate'] = datetime.now()
                
                messagebox.showinfo("Succès", f"Rapport complet exporté:\n{filename}")
            except Exception as e:
                messagebox.showerror("Erreur", f"Erreur lors de l'export:\n{str(e)}")


def main():
    """Fonction principale"""
    root = tk.Tk()
    app = PunchingShearGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()