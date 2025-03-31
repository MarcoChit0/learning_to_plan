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
	instrument7 - instrument
	satellite2 - satellite
	instrument8 - instrument
	instrument9 - instrument
	instrument10 - instrument
	instrument11 - instrument
	satellite3 - satellite
	instrument12 - instrument
	thermograph0 - mode
	groundstation3 - direction
	star0 - direction
	star4 - direction
	groundstation1 - direction
	groundstation2 - direction
	star5 - direction
	planet6 - direction
	phenomenon7 - direction
	planet8 - direction
)
(:init
	(supports instrument0 thermograph0)
	(calibration_target instrument0 groundstation3)
	(supports instrument1 thermograph0)
	(calibration_target instrument1 star4)
	(supports instrument2 thermograph0)
	(calibration_target instrument2 star4)
	(on_board instrument0 satellite0)
	(on_board instrument1 satellite0)
	(on_board instrument2 satellite0)
	(power_avail satellite0)
	(pointing satellite0 star5)
	(supports instrument3 thermograph0)
	(calibration_target instrument3 star0)
	(supports instrument4 thermograph0)
	(calibration_target instrument4 star4)
	(supports instrument5 thermograph0)
	(calibration_target instrument5 star0)
	(supports instrument6 thermograph0)
	(calibration_target instrument6 star4)
	(supports instrument7 thermograph0)
	(calibration_target instrument7 star0)
	(on_board instrument3 satellite1)
	(on_board instrument4 satellite1)
	(on_board instrument5 satellite1)
	(on_board instrument6 satellite1)
	(on_board instrument7 satellite1)
	(power_avail satellite1)
	(pointing satellite1 phenomenon7)
	(supports instrument8 thermograph0)
	(calibration_target instrument8 star4)
	(supports instrument9 thermograph0)
	(calibration_target instrument9 groundstation1)
	(supports instrument10 thermograph0)
	(calibration_target instrument10 star4)
	(supports instrument11 thermograph0)
	(calibration_target instrument11 groundstation1)
	(on_board instrument8 satellite2)
	(on_board instrument9 satellite2)
	(on_board instrument10 satellite2)
	(on_board instrument11 satellite2)
	(power_avail satellite2)
	(pointing satellite2 groundstation3)
	(supports instrument12 thermograph0)
	(calibration_target instrument12 groundstation2)
	(on_board instrument12 satellite3)
	(power_avail satellite3)
	(pointing satellite3 planet6)
)
(:goal (and
	(pointing satellite1 phenomenon7)
	(have_image star5 thermograph0)
	(have_image planet6 thermograph0)
	(have_image phenomenon7 thermograph0)
	(have_image planet8 thermograph0)
))

)
