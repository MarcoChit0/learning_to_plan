(define (problem strips-sat-x-1)
(:domain satellite)
(:objects
	satellite0 - satellite
	instrument0 - instrument
	instrument1 - instrument
	instrument2 - instrument
	instrument3 - instrument
	instrument4 - instrument
	satellite1 - satellite
	instrument5 - instrument
	infrared1 - mode
	image2 - mode
	thermograph0 - mode
	star0 - direction
	phenomenon1 - direction
	star2 - direction
)
(:init
	(supports instrument0 infrared1)
	(calibration_target instrument0 star0)
	(supports instrument1 infrared1)
	(supports instrument1 thermograph0)
	(supports instrument1 image2)
	(calibration_target instrument1 star0)
	(supports instrument2 infrared1)
	(supports instrument2 image2)
	(calibration_target instrument2 star0)
	(supports instrument3 image2)
	(supports instrument3 thermograph0)
	(calibration_target instrument3 star0)
	(supports instrument4 infrared1)
	(calibration_target instrument4 star0)
	(on_board instrument0 satellite0)
	(on_board instrument1 satellite0)
	(on_board instrument2 satellite0)
	(on_board instrument3 satellite0)
	(on_board instrument4 satellite0)
	(power_avail satellite0)
	(pointing satellite0 star0)
	(supports instrument5 thermograph0)
	(supports instrument5 infrared1)
	(supports instrument5 image2)
	(calibration_target instrument5 star0)
	(on_board instrument5 satellite1)
	(power_avail satellite1)
	(pointing satellite1 star0)
)
(:goal (and
	(pointing satellite1 phenomenon1)
	(have_image phenomenon1 infrared1)
	(have_image star2 image2)
))

)
