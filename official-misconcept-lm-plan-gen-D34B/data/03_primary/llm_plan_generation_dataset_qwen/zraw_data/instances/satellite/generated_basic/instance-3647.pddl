(define (problem strips-sat-x-1)
(:domain satellite)
(:objects
	satellite0 - satellite
	instrument0 - instrument
	satellite1 - satellite
	instrument1 - instrument
	instrument2 - instrument
	instrument3 - instrument
	instrument4 - instrument
	instrument5 - instrument
	satellite2 - satellite
	instrument6 - instrument
	instrument7 - instrument
	instrument8 - instrument
	instrument9 - instrument
	satellite3 - satellite
	instrument10 - instrument
	instrument11 - instrument
	instrument12 - instrument
	image1 - mode
	thermograph0 - mode
	star0 - direction
	star1 - direction
	groundstation2 - direction
	planet3 - direction
	star4 - direction
)
(:init
	(supports instrument0 image1)
	(calibration_target instrument0 star0)
	(on_board instrument0 satellite0)
	(power_avail satellite0)
	(pointing satellite0 groundstation2)
	(supports instrument1 image1)
	(calibration_target instrument1 star1)
	(supports instrument2 image1)
	(supports instrument2 thermograph0)
	(calibration_target instrument2 star1)
	(supports instrument3 image1)
	(supports instrument3 thermograph0)
	(calibration_target instrument3 groundstation2)
	(supports instrument4 thermograph0)
	(supports instrument4 image1)
	(calibration_target instrument4 star1)
	(supports instrument5 image1)
	(supports instrument5 thermograph0)
	(calibration_target instrument5 star1)
	(on_board instrument1 satellite1)
	(on_board instrument2 satellite1)
	(on_board instrument3 satellite1)
	(on_board instrument4 satellite1)
	(on_board instrument5 satellite1)
	(power_avail satellite1)
	(pointing satellite1 star4)
	(supports instrument6 image1)
	(supports instrument6 thermograph0)
	(calibration_target instrument6 groundstation2)
	(supports instrument7 thermograph0)
	(calibration_target instrument7 star1)
	(supports instrument8 image1)
	(supports instrument8 thermograph0)
	(calibration_target instrument8 star1)
	(supports instrument9 thermograph0)
	(supports instrument9 image1)
	(calibration_target instrument9 star1)
	(on_board instrument6 satellite2)
	(on_board instrument7 satellite2)
	(on_board instrument8 satellite2)
	(on_board instrument9 satellite2)
	(power_avail satellite2)
	(pointing satellite2 groundstation2)
	(supports instrument10 thermograph0)
	(supports instrument10 image1)
	(calibration_target instrument10 star1)
	(supports instrument11 thermograph0)
	(calibration_target instrument11 groundstation2)
	(supports instrument12 thermograph0)
	(supports instrument12 image1)
	(calibration_target instrument12 groundstation2)
	(on_board instrument10 satellite3)
	(on_board instrument11 satellite3)
	(on_board instrument12 satellite3)
	(power_avail satellite3)
	(pointing satellite3 star0)
)
(:goal (and
	(pointing satellite1 star1)
	(pointing satellite2 planet3)
	(have_image planet3 thermograph0)
	(have_image star4 thermograph0)
))

)
