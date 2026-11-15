import { useState, useEffect, useRef, useCallback } from "react";

export const CAMERA_STATUS = {
  INITIALIZING: "INITIALIZING",
  REQUESTING_PERMISSION: "REQUESTING_PERMISSION",
  ACTIVE: "ACTIVE",
  STOPPED: "STOPPED",
  PERMISSION_DENIED: "PERMISSION_DENIED",
  NOT_AVAILABLE: "NOT_AVAILABLE",
  ERROR: "ERROR",
};

export const CAMERA_STATUS_LABELS = {
  [CAMERA_STATUS.INITIALIZING]: "Initializing Camera",
  [CAMERA_STATUS.REQUESTING_PERMISSION]: "Requesting Permission",
  [CAMERA_STATUS.ACTIVE]: "Camera Active",
  [CAMERA_STATUS.STOPPED]: "Camera Stopped",
  [CAMERA_STATUS.PERMISSION_DENIED]: "Camera Permission Denied",
  [CAMERA_STATUS.NOT_AVAILABLE]: "Camera Not Available",
  [CAMERA_STATUS.ERROR]: "Camera Error",
};

export function useWebcam({
  autoStart = true,
  preferredWidth = 1280,
  preferredHeight = 720,
} = {}) {
  const videoRef = useRef(null);
  const streamRef = useRef(null);
  const isMountedRef = useRef(true);
  const selectedDeviceIdRef = useRef("");

  const [status, setStatus] = useState(CAMERA_STATUS.INITIALIZING);
  const [stream, setStream] = useState(null);
  const [errorMessage, setErrorMessage] = useState(null);
  const [devices, setDevices] = useState([]);
  const [selectedDeviceId, setSelectedDeviceId] = useState("");

  useEffect(() => {
    selectedDeviceIdRef.current = selectedDeviceId;
  }, [selectedDeviceId]);

  // Helper to enumerate available video devices without triggering stream re-acquisitions
  const updateDevices = useCallback(async () => {
    try {
      if (!navigator.mediaDevices?.enumerateDevices) return;
      const allDevices = await navigator.mediaDevices.enumerateDevices();
      const videoInputs = allDevices.filter((d) => d.kind === "videoinput");
      if (isMountedRef.current) {
        setDevices(videoInputs);
        if (videoInputs.length > 0 && !selectedDeviceIdRef.current) {
          selectedDeviceIdRef.current = videoInputs[0].deviceId;
          setSelectedDeviceId(videoInputs[0].deviceId);
        }
      }
    } catch {
      // Non-fatal device enumeration error
    }
  }, []);

  // Stop camera tracks cleanly
  const stopCamera = useCallback(() => {
    if (streamRef.current) {
      streamRef.current.getTracks().forEach((track) => {
        try {
          track.onended = null;
          track.stop();
        } catch {
          // ignore already stopped tracks
        }
      });
      streamRef.current = null;
    }

    if (videoRef.current) {
      videoRef.current.srcObject = null;
    }

    if (isMountedRef.current) {
      setStream(null);
      setStatus(CAMERA_STATUS.STOPPED);
      setErrorMessage(null);
    }
  }, []);

  // Start camera stream with optimal resolution & fallback
  const startCamera = useCallback(
    async (deviceIdToUse) => {
      const targetDeviceId = deviceIdToUse || selectedDeviceIdRef.current;

      // Clean up any existing stream first to avoid duplicate streams
      if (streamRef.current) {
        streamRef.current.getTracks().forEach((track) => {
          track.onended = null;
          track.stop();
        });
        streamRef.current = null;
      }
      if (videoRef.current) {
        videoRef.current.srcObject = null;
      }

      // Small hardware unbind pause so Windows camera driver releases exclusive lock
      await new Promise((resolve) => setTimeout(resolve, 80));

      if (isMountedRef.current) {
        setStatus(CAMERA_STATUS.REQUESTING_PERMISSION);
        setErrorMessage(null);
      }

      if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
        if (isMountedRef.current) {
          setStatus(CAMERA_STATUS.NOT_AVAILABLE);
          setErrorMessage("Browser does not support camera access (mediaDevices.getUserMedia).");
        }
        return;
      }

      // Multi-tier constraints ladder from optimal to most permissive
      const constraintTiers = [];

      // Tier 1: User's explicitly chosen device (ideal, not exact, to avoid NotFoundError aborts)
      if (targetDeviceId) {
        constraintTiers.push({
          audio: false,
          video: {
            deviceId: { ideal: targetDeviceId },
            width: { ideal: preferredWidth },
            height: { ideal: preferredHeight },
          },
        });
        constraintTiers.push({
          audio: false,
          video: {
            deviceId: { ideal: targetDeviceId },
          },
        });
      }

      // Tier 2: Optimal resolution (ideal, no facingMode which can fail on Windows desktop webcams)
      constraintTiers.push({
        audio: false,
        video: {
          width: { ideal: preferredWidth },
          height: { ideal: preferredHeight },
        },
      });

      // Tier 3: Standard 640x480 resolution
      constraintTiers.push({
        audio: false,
        video: {
          width: { ideal: 640 },
          height: { ideal: 480 },
        },
      });

      // Tier 4: Most permissive basic video constraint
      constraintTiers.push({
        audio: false,
        video: true,
      });

      try {
        let activeStream = null;
        let lastError = null;

        for (const tierConstraints of constraintTiers) {
          try {
            activeStream = await navigator.mediaDevices.getUserMedia(tierConstraints);
            if (activeStream) break;
          } catch (tierErr) {
            lastError = tierErr;
            // Only abort early if permission was explicitly denied by the user
            if (
              tierErr.name === "NotAllowedError" ||
              tierErr.name === "PermissionDeniedError"
            ) {
              throw tierErr;
            }
            // For any other error (OverconstrainedError, NotFoundError on constraint, NotReadableError),
            // continue down the ladder to try simpler, more permissive constraints
          }
        }

        if (!activeStream && lastError) {
          throw lastError;
        }

        if (!isMountedRef.current) {
          // Component unmounted while waiting for user prompt
          activeStream.getTracks().forEach((t) => t.stop());
          return;
        }

        // Attach disconnection handler on active video track
        const videoTracks = activeStream.getVideoTracks
          ? activeStream.getVideoTracks()
          : (activeStream.getTracks ? activeStream.getTracks().filter((t) => t.kind === "video") : []);
        const videoTrack = videoTracks[0];
        if (videoTrack) {
          videoTrack.onended = () => {
            if (isMountedRef.current) {
              if (videoRef.current) {
                videoRef.current.srcObject = null;
              }
              streamRef.current = null;
              setStream(null);
              setStatus(CAMERA_STATUS.ERROR);
              setErrorMessage("Camera was disconnected. Please reconnect your video device and click Retry.");
            }
          };
        }

        streamRef.current = activeStream;

        if (videoRef.current) {
          videoRef.current.srcObject = activeStream;
        }

        setStream(activeStream);
        setStatus(CAMERA_STATUS.ACTIVE);
        setErrorMessage(null);
        await updateDevices();
      } catch (err) {
        if (!isMountedRef.current) return;

        // Reset target device if specific device failed
        if (targetDeviceId) {
          selectedDeviceIdRef.current = "";
          setSelectedDeviceId("");
        }

        if (err.name === "NotAllowedError" || err.name === "PermissionDeniedError") {
          setStatus(CAMERA_STATUS.PERMISSION_DENIED);
          setErrorMessage(
            "Camera access denied. Please click the camera or lock icon in your browser address bar to allow camera access, then click Retry."
          );
        } else if (err.name === "NotFoundError" || err.name === "DevicesNotFoundError") {
          setStatus(CAMERA_STATUS.NOT_AVAILABLE);
          setErrorMessage(
            "No webcam detected. If your camera is plugged in or built-in, please close other tabs (e.g. http://127.0.0.1:5173) or apps holding the camera, check Windows Camera Settings, and click Retry."
          );
        } else if (
          err.name === "NotReadableError" ||
          err.name === "TrackStartError" ||
          (err.message && err.message.toLowerCase().includes("could not start video source"))
        ) {
          setStatus(CAMERA_STATUS.ERROR);
          setErrorMessage(
            "Could not start video source. Your camera is in use by another application (e.g. Zoom, Teams, WhatsApp, or another browser tab). Close other tabs/apps and click Retry."
          );
        } else {
          setStatus(CAMERA_STATUS.ERROR);
          setErrorMessage(err.message || "Failed to start camera preview.");
        }
      }
    },
    [preferredWidth, preferredHeight, updateDevices]
  );

  // Switch camera when multiple devices exist
  const switchCamera = useCallback(
    async (newDeviceId) => {
      selectedDeviceIdRef.current = newDeviceId;
      setSelectedDeviceId(newDeviceId);
      await startCamera(newDeviceId);
    },
    [startCamera]
  );

  // Device change listener for hardware connect / disconnect events
  useEffect(() => {
    if (!navigator.mediaDevices?.addEventListener) return;

    const handleDeviceChange = async () => {
      await updateDevices();
      if (streamRef.current && isMountedRef.current) {
        const tracks = streamRef.current.getVideoTracks
          ? streamRef.current.getVideoTracks()
          : (streamRef.current.getTracks ? streamRef.current.getTracks().filter((t) => t.kind === "video") : []);
        const track = tracks[0];
        if (!track || track.readyState === "ended") {
          if (videoRef.current) {
            videoRef.current.srcObject = null;
          }
          streamRef.current = null;
          setStream(null);
          setStatus(CAMERA_STATUS.ERROR);
          setErrorMessage("Camera was disconnected. Please reconnect your video device and click Retry.");
        }
      }
    };

    navigator.mediaDevices.addEventListener("devicechange", handleDeviceChange);
    return () => {
      navigator.mediaDevices.removeEventListener("devicechange", handleDeviceChange);
    };
  }, [updateDevices]);

  // Auto-start on mount, clean up on unmount
  useEffect(() => {
    isMountedRef.current = true;
    if (autoStart) {
      startCamera();
    } else {
      setStatus(CAMERA_STATUS.STOPPED);
    }

    return () => {
      isMountedRef.current = false;
      // Stop all tracks when user navigates away
      if (streamRef.current) {
        streamRef.current.getTracks().forEach((t) => {
          t.onended = null;
          t.stop();
        });
        streamRef.current = null;
      }
      if (videoRef.current) {
        videoRef.current.srcObject = null;
      }
    };
  }, [autoStart, startCamera]);

  return {
    videoRef,
    stream,
    status,
    statusLabel: CAMERA_STATUS_LABELS[status] || "Camera Status",
    isActive: status === CAMERA_STATUS.ACTIVE,
    isInitializing:
      status === CAMERA_STATUS.INITIALIZING ||
      status === CAMERA_STATUS.REQUESTING_PERMISSION,
    errorMessage,
    devices,
    selectedDeviceId,
    startCamera,
    stopCamera,
    switchCamera,
  };
}
