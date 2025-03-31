(define (problem strips-sat-x-1)
(:domain satellite)
(:objects
	satellite0 - satellite
	instrument0 - instrument
	instrument1 - instrument
	instrument2 - instrument
	satellite1 - satellite
	instrument3 - instrument
	satellite2 - satellite
	instrument4 - instrument
	image1 - mode
	thermograph0 - mode
	image2 - mode
	star1 - direction
	star0 - direction
	star2 - direction
	phenomenon3 - direction
	phenomenon4 - direction
	phenomenon5 - direction
	star6 - direction
)
(:init
	(supports instrument0 image2)
	(supports instrument0 thermograph0)
	(calibration_target instrument0 star0)
	(supports instrument1 image2)
	(supports instrument1 thermograph0)
	(calibration_target instrument1 star1)
	(supports instrument2 image2)
	(calibration_target instrument2 star1)
	(on_board instrument0 satellite0)
	(on_board instrument1 satellite0)
	(on_board instrument2 satellite0)
	(power_avail satellite0)
	(pointing satellite0 phenomenon3)
	(supports instrument3 image2)
	(supports instrument3 image1)
	(supports instrument3 thermograph0)
	(calibration_target instrument3 star1)
	(on_board instrument3 satellite1)
	(power_avail satellite1)
	(pointing satellite1 phenomenon5)
	(supports instrument4 image1)
	(supports instrument4 thermograph0)
	(supports instrument4 image2)
	(calibration_target instrument4 star0)
	(on_board instrument4 satellite2)
	(power_avail satellite2)
	(pointing satellite2 star0)
)
(:goal (and
	(pointing satellite0 star0)
	(pointing satellite2 phenomenon5)
	(have_image star2 thermograph0)
	(have_image phenomenon3 image2)
	(have_image phenomenon4 image2)
	(have_image phenomenon5 thermograph0)
	(have_image star6 image1)
))

)
