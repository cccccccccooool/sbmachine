package main

import "testing"

func TestValidateEntitySnapshots(t *testing.T) {
	tests := []struct {
		name                   string
		tickCount, rosterCount int
		wantErr                bool
	}{
		{name: "complete", tickCount: 1, rosterCount: 1},
		{name: "empty ticks", tickCount: 0, rosterCount: 1, wantErr: true},
		{name: "empty roster", tickCount: 1, rosterCount: 0, wantErr: true},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			err := validateEntitySnapshots(tt.tickCount, tt.rosterCount)
			if (err != nil) != tt.wantErr {
				t.Fatalf("validateEntitySnapshots(%d, %d) error = %v, wantErr %v", tt.tickCount, tt.rosterCount, err, tt.wantErr)
			}
		})
	}
}
