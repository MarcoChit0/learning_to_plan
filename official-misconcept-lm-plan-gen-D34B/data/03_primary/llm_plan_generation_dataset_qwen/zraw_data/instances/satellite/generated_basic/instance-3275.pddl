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
	instrument5 - instrument
	satellite2 - satellite
	instrument6 - instrument
	instrument7 - instrument
	satellite3 - satellite
	instrument8 - instrument
	instrument9 - instrument
	thermograph2 - mode
	image0 - mode
	image1 - mode
	star0 - direction
	star1 - direction
	star2 - direction
	planet3 - direction
)
(:init
	(supports instrument0 thermograph2)
	(calibration_target instrument0 star0)
	(supports instrument1 image0)
	(calibration_target instrument1 star0)
	(supports instrument2 image1)
	(supports instrument2 image0)
	(supports instrument2 thermograph2)
	(calibration_target instrument2 star2)
	(on_board instrument0 satellite0)
	(on_board instrument1 satellite0)
	(on_board instrument2 satellite0)
	(power_avail satellite0)
	(pointing satellite0 planet3)
	(supports instrument3 image1)
	(supports instrument3 thermograph2)
	(supports instrument3 image0)
	(calibration_target instrument3 star0)
	(supports instrument4 image0)
	(supports instrument4 image1)
	(calibration_target instrument4 star1)
	(supports instrument5 thermograph2)
	(supports instrument5 image0)
	(calibration_target instrument5 star1)
	(on_board instrument3 satellite1)
	(on_board instrument4 satellite1)
	(on_board instrument5 satellite1)
	(power_avail satellite1)
	(pointing satellite1 star0)
	(supports instrument6 image0)
	(calibration_target instrument6 star1)
	(supports instrument7 thermograph2)
	(supports instrument7 image1)
	(calibration_target instrument7 star2)
	(on_board instrument6 satellite2)
	(on_board instrument7 satellite2)
	(power_avail satellite2)
	(pointing satellite2 star2)
	(supports instrument8 thermograph2)
	(calibration_target instrument8 star1)
	(supports instrument9 image1)
	(supports instrument9 image0)
	(calibration_target instrument9 star2)
	(on_board instrument8 satellite3)
	(on_board instrument9 satellite3)
	(power_avail satellite3)
	(pointing satellite3 star2)
)
(:goal (and
	(pointing satellite1 star0)
	(have_image planet3 image0)
))

)
