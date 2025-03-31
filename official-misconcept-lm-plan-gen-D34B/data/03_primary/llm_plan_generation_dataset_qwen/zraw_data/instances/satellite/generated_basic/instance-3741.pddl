(define (problem strips-sat-x-1)
(:domain satellite)
(:objects
	satellite0 - satellite
	instrument0 - instrument
	instrument1 - instrument
	satellite1 - satellite
	instrument2 - instrument
	satellite2 - satellite
	instrument3 - instrument
	instrument4 - instrument
	instrument5 - instrument
	satellite3 - satellite
	instrument6 - instrument
	satellite4 - satellite
	instrument7 - instrument
	instrument8 - instrument
	instrument9 - instrument
	satellite5 - satellite
	instrument10 - instrument
	instrument11 - instrument
	instrument12 - instrument
	instrument13 - instrument
	instrument14 - instrument
	instrument15 - instrument
	thermograph1 - mode
	thermograph0 - mode
	image2 - mode
	star2 - direction
	groundstation1 - direction
	star0 - direction
	planet3 - direction
)
(:init
	(supports instrument0 image2)
	(calibration_target instrument0 star2)
	(supports instrument1 thermograph0)
	(supports instrument1 image2)
	(calibration_target instrument1 groundstation1)
	(on_board instrument0 satellite0)
	(on_board instrument1 satellite0)
	(power_avail satellite0)
	(pointing satellite0 star0)
	(supports instrument2 image2)
	(calibration_target instrument2 star2)
	(on_board instrument2 satellite1)
	(power_avail satellite1)
	(pointing satellite1 groundstation1)
	(supports instrument3 thermograph1)
	(supports instrument3 image2)
	(calibration_target instrument3 star2)
	(supports instrument4 image2)
	(supports instrument4 thermograph0)
	(supports instrument4 thermograph1)
	(calibration_target instrument4 groundstation1)
	(supports instrument5 thermograph1)
	(supports instrument5 thermograph0)
	(calibration_target instrument5 star2)
	(on_board instrument3 satellite2)
	(on_board instrument4 satellite2)
	(on_board instrument5 satellite2)
	(power_avail satellite2)
	(pointing satellite2 planet3)
	(supports instrument6 image2)
	(supports instrument6 thermograph0)
	(calibration_target instrument6 groundstation1)
	(on_board instrument6 satellite3)
	(power_avail satellite3)
	(pointing satellite3 planet3)
	(supports instrument7 image2)
	(calibration_target instrument7 groundstation1)
	(supports instrument8 image2)
	(supports instrument8 thermograph1)
	(supports instrument8 thermograph0)
	(calibration_target instrument8 star2)
	(supports instrument9 thermograph0)
	(supports instrument9 image2)
	(calibration_target instrument9 groundstation1)
	(on_board instrument7 satellite4)
	(on_board instrument8 satellite4)
	(on_board instrument9 satellite4)
	(power_avail satellite4)
	(pointing satellite4 star2)
	(supports instrument10 thermograph0)
	(supports instrument10 image2)
	(calibration_target instrument10 star2)
	(supports instrument11 thermograph1)
	(supports instrument11 thermograph0)
	(supports instrument11 image2)
	(calibration_target instrument11 groundstation1)
	(supports instrument12 thermograph1)
	(supports instrument12 thermograph0)
	(supports instrument12 image2)
	(calibration_target instrument12 star0)
	(supports instrument13 thermograph1)
	(supports instrument13 thermograph0)
	(supports instrument13 image2)
	(calibration_target instrument13 groundstation1)
	(supports instrument14 thermograph1)
	(supports instrument14 image2)
	(supports instrument14 thermograph0)
	(calibration_target instrument14 groundstation1)
	(supports instrument15 thermograph0)
	(supports instrument15 thermograph1)
	(supports instrument15 image2)
	(calibration_target instrument15 star0)
	(on_board instrument10 satellite5)
	(on_board instrument11 satellite5)
	(on_board instrument12 satellite5)
	(on_board instrument13 satellite5)
	(on_board instrument14 satellite5)
	(on_board instrument15 satellite5)
	(power_avail satellite5)
	(pointing satellite5 star2)
)
(:goal (and
	(pointing satellite1 star2)
	(pointing satellite2 star2)
	(pointing satellite3 star0)
	(pointing satellite5 planet3)
	(have_image planet3 thermograph0)
))

)
