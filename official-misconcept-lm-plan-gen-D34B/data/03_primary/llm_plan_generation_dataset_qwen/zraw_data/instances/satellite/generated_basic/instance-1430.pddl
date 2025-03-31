(define (problem strips-sat-x-1)
(:domain satellite)
(:objects
	satellite0 - satellite
	instrument0 - instrument
	satellite1 - satellite
	instrument1 - instrument
	satellite2 - satellite
	instrument2 - instrument
	infrared3 - mode
	infrared1 - mode
	image0 - mode
	image2 - mode
	star0 - direction
	phenomenon1 - direction
	planet2 - direction
	phenomenon3 - direction
)
(:init
	(supports instrument0 infrared1)
	(supports instrument0 image0)
	(supports instrument0 infrared3)
	(calibration_target instrument0 star0)
	(on_board instrument0 satellite0)
	(power_avail satellite0)
	(pointing satellite0 phenomenon1)
	(supports instrument1 infrared1)
	(supports instrument1 image0)
	(calibration_target instrument1 star0)
	(on_board instrument1 satellite1)
	(power_avail satellite1)
	(pointing satellite1 planet2)
	(supports instrument2 image2)
	(supports instrument2 infrared1)
	(supports instrument2 infrared3)
	(calibration_target instrument2 star0)
	(on_board instrument2 satellite2)
	(power_avail satellite2)
	(pointing satellite2 planet2)
)
(:goal (and
	(pointing satellite0 phenomenon1)
	(pointing satellite1 star0)
	(have_image phenomenon1 infrared3)
	(have_image planet2 image0)
	(have_image phenomenon3 infrared1)
))

)
