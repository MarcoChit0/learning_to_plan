(define (problem strips-sat-x-1)
(:domain satellite)
(:objects
	satellite0 - satellite
	instrument0 - instrument
	satellite1 - satellite
	instrument1 - instrument
	instrument2 - instrument
	image2 - mode
	infrared0 - mode
	thermograph1 - mode
	star0 - direction
	groundstation1 - direction
	phenomenon2 - direction
	star3 - direction
	planet4 - direction
	phenomenon5 - direction
)
(:init
	(supports instrument0 image2)
	(calibration_target instrument0 groundstation1)
	(on_board instrument0 satellite0)
	(power_avail satellite0)
	(pointing satellite0 star0)
	(supports instrument1 thermograph1)
	(supports instrument1 infrared0)
	(supports instrument1 image2)
	(calibration_target instrument1 groundstation1)
	(supports instrument2 infrared0)
	(calibration_target instrument2 groundstation1)
	(on_board instrument1 satellite1)
	(on_board instrument2 satellite1)
	(power_avail satellite1)
	(pointing satellite1 groundstation1)
)
(:goal (and
	(have_image phenomenon2 image2)
	(have_image star3 infrared0)
	(have_image planet4 thermograph1)
	(have_image phenomenon5 thermograph1)
))

)
