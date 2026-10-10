package proxias

// ControlPlaneOption configures a ControlPlane. NewControlPlane applies them, but none is defined yet.
type ControlPlaneOption func(*controlPlane)
