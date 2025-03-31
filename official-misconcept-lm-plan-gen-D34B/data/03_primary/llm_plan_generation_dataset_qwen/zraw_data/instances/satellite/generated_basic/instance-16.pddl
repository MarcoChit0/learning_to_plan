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
	instrument6 - instrument
	satellite2 - satellite
	instrument7 - instrument
	instrument8 - instrument
	instrument9 - instrument
	instrument10 - instrument
	instrument11 - instrument
	thermograph0 - mode
	image1 - mode
	star0 - direction
	phenomenon1 - direction
	star2 - direction
	phenomenon3 - direction
	planet4 - direction
	planet5 - direction
)
(:init
	(supports instrument0 image1)
	(calibration_target instrument0 star0)
	(supports instrument1 thermograph0)
	(calibration_target instrument1 star0)
	(supports instrument2 thermograph0)
	(supports instrument2 image1)
	(calibration_target instrument2 star0)
	(on_board instrument0 satellite0)
	(on_board instrument1 satellite0)
	(on_board instrument2 satellite0)
	(power_avail satellite0)
	(pointing satellite0 star2)
	(supports instrument3 thermograph0)
	(supports instrument3 image1)
	(calibration_target instrument3 star0)
	(supports instrument4 image1)
	(supports instrument4 thermograph0)
	(calibration_target instrument4 star0)
	(supports instrument5 image1)
	(supports instrument5 thermograph0)
	(calibration_target instrument5 star0)
	(supports instrument6 thermograph0)
	(supports instrument6 image1)
	(calibration_target instrument6 star0)
	(on_board instrument3 satellite1)
	(on_board instrument4 satellite1)
	(on_board instrument5 satellite1)
	(on_board instrument6 satellite1)
	(power_avail satellite1)
	(pointing satellite1 phenomenon1)
	(supports instrument7 thermograph0)
	(calibration_target instrument7 star0)
	(supports instrument8 image1)
	(supports instrument8 thermograph0)
	(calibration_target instrument8 star0)
	(supports instrument9 image1)
	(calibration_target instrument9 star0)
	(supports instrument10 thermograph0)
	(supports instrument10 image1)
	(calibration_target instrument10 star0)
	(supports instrument11 image1)
	(supports instrument11 thermograph0)
	(calibration_target instrument11 star0)
	(on_board instrument7 satellite2)
	(on_board instrument8 satellite2)
	(on_board instrument9 satellite2)
	(on_board instrument10 satellite2)
	(on_board instrument11 satellite2)
	(power_avail satellite2)
	(pointing satellite2 star0)
)
(:goal (and
	(pointing satellite2 star0)
	(have_image phenomenon1 image1)
	(have_image star2 thermograph0)
	(have_image phenomenon3 thermograph0)
	(have_image planet4 image1)
	(have_image planet5 image1)
))

)
