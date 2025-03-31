(define (problem strips-sat-x-1)
(:domain satellite)
(:objects
	satellite0 - satellite
	instrument0 - instrument
	satellite1 - satellite
	instrument1 - instrument
	infrared0 - mode
	image1 - mode
	star0 - direction
	groundstation1 - direction
	planet2 - direction
	phenomenon3 - direction
)
(:init
	(supports instrument0 image1)
	(supports instrument0 infrared0)
	(calibration_target instrument0 star0)
	(on_board instrument0 satellite0)
	(power_avail satellite0)
	(pointing satellite0 star0)
	(supports instrument1 image1)
	(calibration_target instrument1 groundstation1)
	(on_board instrument1 satellite1)
	(power_avail satellite1)
	(pointing satellite1 groundstation1)
)
(:goal (and
	(pointing satellite0 phenomenon3)
	(pointing satellite1 groundstation1)
	(have_image planet2 infrared0)
	(have_image phenomenon3 infrared0)
))

)
