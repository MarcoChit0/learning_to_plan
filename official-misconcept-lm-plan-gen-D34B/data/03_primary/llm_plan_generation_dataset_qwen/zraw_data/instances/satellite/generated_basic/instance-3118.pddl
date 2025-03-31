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
	satellite3 - satellite
	instrument5 - instrument
	instrument6 - instrument
	instrument7 - instrument
	satellite4 - satellite
	instrument8 - instrument
	satellite5 - satellite
	instrument9 - instrument
	instrument10 - instrument
	thermograph0 - mode
	star0 - direction
	groundstation1 - direction
	star2 - direction
	star3 - direction
	planet4 - direction
)
(:init
	(supports instrument0 thermograph0)
	(calibration_target instrument0 star0)
	(supports instrument1 thermograph0)
	(calibration_target instrument1 star0)
	(on_board instrument0 satellite0)
	(on_board instrument1 satellite0)
	(power_avail satellite0)
	(pointing satellite0 star3)
	(supports instrument2 thermograph0)
	(calibration_target instrument2 star0)
	(supports instrument3 thermograph0)
	(calibration_target instrument3 star0)
	(on_board instrument2 satellite1)
	(on_board instrument3 satellite1)
	(power_avail satellite1)
	(pointing satellite1 groundstation1)
	(supports instrument4 thermograph0)
	(calibration_target instrument4 groundstation1)
	(on_board instrument4 satellite2)
	(power_avail satellite2)
	(pointing satellite2 star2)
	(supports instrument5 thermograph0)
	(calibration_target instrument5 star0)
	(supports instrument6 thermograph0)
	(calibration_target instrument6 star0)
	(supports instrument7 thermograph0)
	(calibration_target instrument7 star0)
	(on_board instrument5 satellite3)
	(on_board instrument6 satellite3)
	(on_board instrument7 satellite3)
	(power_avail satellite3)
	(pointing satellite3 planet4)
	(supports instrument8 thermograph0)
	(calibration_target instrument8 star0)
	(on_board instrument8 satellite4)
	(power_avail satellite4)
	(pointing satellite4 groundstation1)
	(supports instrument9 thermograph0)
	(calibration_target instrument9 star0)
	(supports instrument10 thermograph0)
	(calibration_target instrument10 groundstation1)
	(on_board instrument9 satellite5)
	(on_board instrument10 satellite5)
	(power_avail satellite5)
	(pointing satellite5 star3)
)
(:goal (and
	(pointing satellite2 star0)
	(have_image star2 thermograph0)
	(have_image star3 thermograph0)
	(have_image planet4 thermograph0)
))

)
