(define (problem strips-sat-x-1)
(:domain satellite)
(:objects
	satellite0 - satellite
	instrument0 - instrument
	instrument1 - instrument
	satellite1 - satellite
	instrument2 - instrument
	instrument3 - instrument
	infrared2 - mode
	thermograph1 - mode
	image0 - mode
	infrared3 - mode
	groundstation0 - direction
	planet1 - direction
)
(:init
	(supports instrument0 infrared3)
	(supports instrument0 infrared2)
	(calibration_target instrument0 groundstation0)
	(supports instrument1 infrared3)
	(calibration_target instrument1 groundstation0)
	(on_board instrument0 satellite0)
	(on_board instrument1 satellite0)
	(power_avail satellite0)
	(pointing satellite0 planet1)
	(supports instrument2 infrared3)
	(calibration_target instrument2 groundstation0)
	(supports instrument3 infrared2)
	(supports instrument3 infrared3)
	(supports instrument3 thermograph1)
	(supports instrument3 image0)
	(calibration_target instrument3 groundstation0)
	(on_board instrument2 satellite1)
	(on_board instrument3 satellite1)
	(power_avail satellite1)
	(pointing satellite1 planet1)
)
(:goal (and
	(pointing satellite1 groundstation0)
	(have_image planet1 infrared3)
))

)
