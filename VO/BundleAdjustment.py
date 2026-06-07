import numpy as np
import gtsam

class GTSAMBundleAdjuster(object):
    """
    Incremental Bundle Adjuster using GTSAM iSAM2 with Delayed Initialization.
    Optimizes both camera poses and 3D landmark positions without using artificial priors.
    """

    def __init__(self, config=None):
        if config is None:
            config = {}

        # 1. Initialize iSAM2
        parameters = gtsam.ISAM2Params()
        # Says if the optimizer should go back and adjust 
        # past variables because the drift is too high
        parameters.setRelinearizeThreshold(0.1)
        parameters.relinearizeSkip = 1
        self.isam2 = gtsam.ISAM2(parameters)

        # 2. Camera Calibration
        fx = float(config["fx"])
        fy = float(config["fy"])
        cx = float(config["cx"])
        cy = float(config["cy"])
        self.calibration = gtsam.Cal3_S2(fx, fy, 0.0, cx, cy)

        # 3. Tuned Noise Models
        # [Roll, Pitch, Yaw, X, Y, Z] - Notice translation (XYZ) has higher uncertainty than rotation
        self.noise_prior = gtsam.noiseModel.Diagonal.Sigmas(
            np.array([0.01, 0.01, 0.01, 0.01, 0.01, 0.01], dtype=float)
        )
        self.noise_odom = gtsam.noiseModel.Diagonal.Sigmas(
            np.array([0.05, 0.05, 0.05, 0.2, 0.2, 0.2], dtype=float)
        )
        # Projection noise (measured in pixels)
        pixel_noise = float(config.get("pixel_noise", 1.0))
        self.noise_proj = gtsam.noiseModel.Isotropic.Sigma(2, pixel_noise)
        
        # --- REDE DE SEGURANÇA GEOMÉTRICA ---
        # Impede que pontos no horizonte (sem paralaxe) explodam a matriz
        self.noise_landmark_prior = gtsam.noiseModel.Isotropic.Sigma(3, 0.05) 

        self.current_key = 0
        self.seen_landmarks = set() # Track which 3D points are FULLY in the graph
        
        # --- SALA DE ESPERA (Delayed Initialization Buffer) ---
        # Armazena a primeira observação de um landmark: 
        # lm_id -> {'pose_symbol': sym, 'u': u, 'v': v, 'initial_3d': (x,y,z)}
        self.pending_landmarks = {} 
        
        self.last_pose = None

    def _pose3_from_rt(self, R, t):
        # creates the pose 3d for the graph
        if isinstance(t, np.ndarray):
            t = np.asarray(t).reshape(3, 1)
            point = gtsam.Point3(float(t[0, 0]), float(t[1, 0]), float(t[2, 0]))
        else:
            point = gtsam.Point3(float(t[0]), float(t[1]), float(t[2]))
        return gtsam.Pose3(gtsam.Rot3(R), point)

    def update(self, absolute_pose, relative_rotation=None, relative_translation=None, 
               observations=None, landmark_initials=None):
        
        if absolute_pose is None:
            raise ValueError("absolute_pose must be provided.")
        
        observations = observations or []
        landmark_initials = landmark_initials or {}

        new_factors = gtsam.NonlinearFactorGraph()
        new_values = gtsam.Values()

        # Add current pose to new values
        R_vo, t_vo = absolute_pose
        current_pose = self._pose3_from_rt(R_vo, t_vo)
        pose_symbol = gtsam.symbol('x', self.current_key)
        new_values.insert(pose_symbol, current_pose)

        # 1. Pose Graph Factors (Odometry & Prior)
        if self.current_key == 0:
            new_factors.add(gtsam.PriorFactorPose3(pose_symbol, current_pose, self.noise_prior))
        else:
            if relative_rotation is not None and relative_translation is not None:
                prev_symbol = gtsam.symbol('x', self.current_key - 1)
                rel_pose = self._pose3_from_rt(relative_rotation, relative_translation)
                new_factors.add(
                    gtsam.BetweenFactorPose3(prev_symbol, pose_symbol, rel_pose, self.noise_odom)
                )

        # 2. Bundle Adjustment Factors (Parallax-Aware Delayed Initialization)
        for lm_id, u, v in observations:
            lm_symbol = gtsam.symbol('l', lm_id)
            measurement = gtsam.Point2(u, v)
            
            if lm_id in self.seen_landmarks:
                # Caso A: O landmark já está maduro e inserido no grafo.
                new_factors.add(
                    gtsam.GenericProjectionFactorCal3_S2(
                        measurement, self.noise_proj, pose_symbol, lm_symbol, self.calibration
                    )
                )
                
            elif lm_id in self.pending_landmarks:
                self.pending_landmarks[lm_id]['obs'].append((pose_symbol, u, v))
                first_t = self.pending_landmarks[lm_id]['first_t']
                distance = np.linalg.norm(t_vo - first_t)
                
                # Exige que o carro ande 0.5m E que o ponto tenha sido rastreado por pelo menos 3 frames
                if distance > 0.01 and len(self.pending_landmarks[lm_id]['obs']) >= 3: 
                    pending_data = self.pending_landmarks.pop(lm_id)
                    
                    # B.1: Inserir a estimativa 3D no grafo
                    lx, ly, lz = pending_data['initial_3d']
                    lm_point = gtsam.Point3(lx, ly, lz)
                    new_values.insert(lm_symbol, lm_point)
                    
                    # Impede o colapso linear em pontos de fuga (movimento puramente frontal)
                    new_factors.add(
                        gtsam.PriorFactorPoint3(lm_symbol, lm_point, self.noise_landmark_prior)
                    )
                    
                    # B.2: Adicionar TODAS as observações acumuladas
                    for obs_pose_sym, obs_u, obs_v in pending_data['obs']:
                        past_measurement = gtsam.Point2(obs_u, obs_v)
                        new_factors.add(
                            gtsam.GenericProjectionFactorCal3_S2(
                                past_measurement, self.noise_proj, obs_pose_sym, lm_symbol, self.calibration
                            )
                        )
                    
                    self.seen_landmarks.add(lm_id)
                    print(f"[BA INFO] Landmark {lm_id} maduro. Baseline: {distance:.2f}m. {len(pending_data['obs'])} observações.")                    
            else:
                # Caso C: Primeira vez que vemos esse ponto.
                if lm_id not in landmark_initials:
                    continue
                    
                # Cria o registro na sala de espera salvando a posição exata (t_vo) da primeira vista
                self.pending_landmarks[lm_id] = {
                    'first_t': t_vo.copy(), 
                    'initial_3d': landmark_initials[lm_id],
                    'obs': [(pose_symbol, u, v)]
                }
        # 3. Update iSAM2 and calculate the estimate (com proteção contra anomalias)
        try:
            # O GTSAM só será atualizado se houver novos fatores além do prior da câmera
            if new_factors.size() > 0:
                self.isam2.update(new_factors, new_values)
                result = self.isam2.calculateEstimate()
                
                optimized_pose = result.atPose3(pose_symbol)
                self.last_pose = optimized_pose
        except Exception as e:
            print(f"[GTSAM INTERNAL ERROR] Falha na otimização da keyframe {self.current_key}: {e}")
        
        self.current_key += 1

        # Retorna a pose otimizada se disponível, caso contrário devolve a odometria bruta
        if self.last_pose is not None:
            return self.last_pose.rotation().matrix(), np.array(
                self.last_pose.translation()
            ).reshape(3, 1)
        else:
            print(f"[BA WARNING] Otimização falhou para keyframe {self.current_key - 1}. Retornando odometria bruta.")
            return R_vo, t_vo

    def get_last_pose(self):
        return self.last_pose