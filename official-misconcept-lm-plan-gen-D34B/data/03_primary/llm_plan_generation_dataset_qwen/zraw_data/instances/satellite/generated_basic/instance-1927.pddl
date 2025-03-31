(define (problem strips-sat-x-1)
(:domain satellite)
(:objects
	satellite0 - satellite
	instrument0 - instrument
	instrument1 - instrument
	satellite1 - satellite
	instrument2 - instrument
	instrument3 - instrument
	satellite2 - satellite
	instrument4 - instrument
	instrument5 - instrument
	satellite3 - satellite
	instrument6 - instrument
	instrument7 - instrument
	instrument8 - instrument
	satellite4 - satellite
	instrument9 - instrument
	instrument10 - instrument
	instrument11 - instrument
	satellite5 - satellite
	instrument12 - instrument
	thermograph0 - mode
	star3 - direction
	star4 - direction
	star0 - direction
	star1 - direction
	star2 - direction
	star5 - direction
	planet6 - direction
)
(:init
	(supports instrument0 thermograph0)
	(calibration_target instrument0 star1)
	(supports instrument1 thermograph0)
	(calibration_target instrument1 star3)
	(on_board instrument0 satellite0)
	(on_board instrument1 satellite0)
	(power_avail satellite0)
	(pointing satellite0 star3)
	(supports instrument2 thermograph0)
	(calibration_target instrument2 star0)
	(supports instrument3 thermograph0)
	(calibration_target instrument3 star2)
	(on_board instrument2 satellite1)
	(on_board instrument3 satellite1)
	(power_avail satellite1)
	(pointing satellite1 star4)
	(supports instrument4 thermograph0)
	(calibration_target instrument4 star4)
	(supports instrument5 thermograph0)
	(calibration_target instrument5 star0)
	(on_board instrument4 satellite2)
	(on_board instrument5 satellite2)
	(power_avail satellite2)
	(pointing satellite2 star4)
	(supports instrument6 thermograph0)
	(calibration_target instrument6 star2)
	(supports instrument7 thermograph0)
	(calibration_target instrument7 star2)
	(supports instrument8 thermograph0)
	(calibration_target instrument8 star0)
	(on_board instrument6 satellite3)
	(on_board instrument7 satellite3)
	(on_board instrument8 satellite3)
	(power_avail satellite3)
	(pointing satellite3 star5)
	(supports instrument9 thermograph0)
	(calibration_target instrument9 star1)
	(supports instrument10 thermograph0)
	(calibration_target instrument10 star2)
	(supports instrument11 thermograph0)
	(calibration_target instrument11 star2)
	(on_board instrument9 satellite4)
	(on_board instrument10 satellite4)
	(on_board instrument11 satellite4)
	(power_avail satellite4)
	(pointing satellite4 star3)
	(supports instrument12 thermograph0)
	(calibration_target instrument12 star2)
	(on_board instrument12 satellite5)
	(power_avail satellite5)
	(pointing satellite5 star0)
)
(:goal (and
	(pointing satellite0 star2)
	(pointing satellite4 planet6)
	(have_image star5 thermograph0)
	(have_image planet6 thermograph0)
))

)
