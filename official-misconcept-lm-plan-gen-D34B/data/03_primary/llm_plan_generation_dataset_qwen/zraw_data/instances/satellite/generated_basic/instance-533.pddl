(define (problem strips-sat-x-1)
(:domain satellite)
(:objects
	satellite0 - satellite
	instrument0 - instrument
	instrument1 - instrument
	instrument2 - instrument
	satellite1 - satellite
	instrument3 - instrument
	instrument4 - instrument
	image0 - mode
	image2 - mode
	thermograph3 - mode
	image1 - mode
	star0 - direction
	groundstation3 - direction
	groundstation1 - direction
	groundstation4 - direction
	star2 - direction
	star5 - direction
	star6 - direction
	phenomenon7 - direction
	phenomenon8 - direction
)
(:init
	(supports instrument0 thermograph3)
	(calibration_target instrument0 groundstation1)
	(supports instrument1 image0)
	(calibration_target instrument1 star2)
	(supports instrument2 image1)
	(supports instrument2 image0)
	(supports instrument2 thermograph3)
	(calibration_target instrument2 groundstation1)
	(on_board instrument0 satellite0)
	(on_board instrument1 satellite0)
	(on_board instrument2 satellite0)
	(power_avail satellite0)
	(pointing satellite0 phenomenon8)
	(supports instrument3 thermograph3)
	(calibration_target instrument3 groundstation4)
	(supports instrument4 image2)
	(calibration_target instrument4 star2)
	(on_board instrument3 satellite1)
	(on_board instrument4 satellite1)
	(power_avail satellite1)
	(pointing satellite1 groundstation4)
)
(:goal (and
	(pointing satellite1 star6)
	(have_image star5 image1)
	(have_image star6 image1)
	(have_image phenomenon7 thermograph3)
	(have_image phenomenon8 image2)
))

)
