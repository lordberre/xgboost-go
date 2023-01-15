package activation

import (
	"github.com/lordberre/xgboost-go/mat"
	"github.com/lordberre/xgboost-go/protobuf"
)

// Activation is an interface that an activation needs to implement.
type Activation interface {
	Transform(rawPrediction mat.Vector) (mat.Vector, error)
	Type() protobuf.ActivateType
	Name() string
}
