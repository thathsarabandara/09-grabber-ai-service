import cv2
import numpy as np
import os
import uuid
from app.core.config import settings


class FaceRecognizer:
    """
    Face recognition engine backed by ChromaDB as a vector database.

    Registration pipeline:
        image → Haar Cascade detection → face crop (100×100)
                → CLAHE normalisation → L2-normalised pixel embedding
                → stored in ChromaDB with operator metadata

    Recognition pipeline (per frame):
        frame → Haar Cascade detection → crop → embedding
              → ChromaDB cosine nearest-neighbour query
              → annotate frame with operator name + confidence
    """

    # Cosine distance threshold for a positive match.
    # ChromaDB cosine distance: 0 = identical, 2 = maximally different.
    MATCH_THRESHOLD = 0.35

    def __init__(self, upload_dir: str = None, vector_db_dir: str = None):
        # ── Resolve directories ────────────────────────────────────────────
        base = "/app" if (os.path.exists("/app") and os.access("/app", os.W_OK)) else os.path.abspath(
            os.path.join(os.path.dirname(__file__), "..", "..")
        )
        # Paths are driven by config.py / docker-compose env vars;
        # constructor args are accepted for testing overrides.
        self.upload_dir = upload_dir or settings.UPLOAD_DIR
        self.vector_db_dir = vector_db_dir or settings.CHROMA_DB_DIR

        os.makedirs(self.upload_dir, exist_ok=True)
        os.makedirs(self.vector_db_dir, exist_ok=True)

        # ── ChromaDB – persistent, embedded (no external server) ──────────
        import chromadb
        self._chroma = chromadb.PersistentClient(path=self.vector_db_dir)
        self._collection = self._chroma.get_or_create_collection(
            name="face_embeddings",
            metadata={"hnsw:space": "cosine"}
        )

        # ── OpenCV Haar Cascade face detector ─────────────────────────────
        self._face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        )

    # ── Embedding ──────────────────────────────────────────────────────────

    def _embed(self, face_gray: np.ndarray) -> list:
        """
        Produce a 10 000-dim L2-normalised embedding from a 100×100 grayscale
        face crop.

        Steps:
          1. CLAHE – equalise lighting across the crop
          2. Flatten to 1-D float32 vector
          3. L2-normalise → unit vector suitable for cosine similarity
        """
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        equalised = clahe.apply(face_gray)
        flat = equalised.astype(np.float32).flatten()
        norm = np.linalg.norm(flat)
        if norm > 0:
            flat /= norm
        return flat.tolist()

    def _detect_largest_face(self, gray: np.ndarray):
        """Return (x, y, w, h) of the largest face in the image, or None."""
        faces = self._face_cascade.detectMultiScale(
            gray, scaleFactor=1.1, minNeighbors=5, minSize=(60, 60)
        )
        if len(faces) == 0:
            return None
        return max(faces, key=lambda f: f[2] * f[3])

    # ── Registration ───────────────────────────────────────────────────────

    def register_face(self, image_bytes: bytes, name: str) -> str:
        """
        Detect the largest face, embed it, persist the vector in ChromaDB,
        and save the crop to disk.

        Returns the saved file path (used as the operator avatar URL).
        Raises Exception if no face is detected or the image is invalid.
        """
        nparr = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if img is None:
            raise Exception("Invalid image data")

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        bbox = self._detect_largest_face(gray)
        if bbox is None:
            return None

        x, y, w, h = bbox
        crop = cv2.resize(gray[y:y + h, x:x + w], (100, 100))

        # Persist crop to disk
        safe_name = name.lower().replace(" ", "-")
        vector_id = uuid.uuid4().hex[:10]
        filename = f"{safe_name}_{vector_id}.jpg"
        save_path = os.path.join(self.upload_dir, filename)
        cv2.imwrite(save_path, crop)

        # Store embedding in ChromaDB
        self._collection.add(
            ids=[vector_id],
            embeddings=[self._embed(crop)],
            metadatas=[{"name": name, "file": filename}]
        )

        return save_path

    # ── Management ─────────────────────────────────────────────────────────

    def remove_operator(self, name: str):
        """Delete all vectors and disk files belonging to an operator."""
        rows = self._collection.get(where={"name": name})
        if rows["ids"]:
            self._collection.delete(ids=rows["ids"])

        safe_name = name.lower().replace(" ", "-")
        if os.path.exists(self.upload_dir):
            for f in os.listdir(self.upload_dir):
                if f.startswith(safe_name + "_"):
                    os.remove(os.path.join(self.upload_dir, f))

    def rename_operator(self, old_name: str, new_name: str):
        """Rename an operator across ChromaDB metadata and disk files."""
        safe_old = old_name.lower().replace(" ", "-")
        safe_new = new_name.lower().replace(" ", "-")

        rows = self._collection.get(where={"name": old_name})
        for vec_id, meta in zip(rows["ids"], rows["metadatas"]):
            new_file = meta["file"].replace(safe_old + "_", safe_new + "_", 1)
            self._collection.update(
                ids=[vec_id],
                metadatas=[{"name": new_name, "file": new_file}]
            )

        if os.path.exists(self.upload_dir):
            for f in os.listdir(self.upload_dir):
                if f.startswith(safe_old + "_"):
                    os.rename(
                        os.path.join(self.upload_dir, f),
                        os.path.join(self.upload_dir, f.replace(safe_old + "_", safe_new + "_", 1))
                    )

    # ── Live frame processing ──────────────────────────────────────────────

    def process_frame(self, frame: np.ndarray, operator_roles: dict = None) -> np.ndarray:
        """
        Detect all faces in the frame, query ChromaDB for the closest stored
        embedding, and draw annotated bounding boxes.

        - Green box  → recognised operator (cosine distance < MATCH_THRESHOLD)
        - Orange box → unknown person
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self._face_cascade.detectMultiScale(
            gray, scaleFactor=1.1, minNeighbors=5, minSize=(60, 60)
        )

        db_count = self._collection.count()

        for (x, y, w, h) in faces:
            crop = cv2.resize(gray[y:y + h, x:x + w], (100, 100))
            embedding = self._embed(crop)

            label = "Unknown"
            color = (0, 165, 255)   # Orange

            if db_count > 0:
                result = self._collection.query(
                    query_embeddings=[embedding],
                    n_results=1,
                    include=["metadatas", "distances"]
                )
                if result["ids"] and result["ids"][0]:
                    distance = result["distances"][0][0]
                    if distance < self.MATCH_THRESHOLD:
                        operator_name = result["metadatas"][0][0]["name"]
                        
                        # Extract role if available
                        role_text = ""
                        if operator_roles and operator_name in operator_roles:
                            access_level = operator_roles[operator_name]
                            if " - " in access_level:
                                access_level = access_level.split(" - ")[1]
                            role_text = f" [{access_level}]"

                        # Convert cosine distance to a human-readable confidence %
                        confidence = int((1.0 - distance / 2.0) * 100)
                        label = f"{operator_name}{role_text} ({confidence}%)"
                        color = (16, 185, 129)  # Green

            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            cv2.putText(
                frame, label,
                (x, max(y - 8, 0)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2
            )

        return frame
