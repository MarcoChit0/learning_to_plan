(define (problem strips-sat-x-1)
(:domain satellite)
(:objects
	satellite0 - satellite
	instrument0 - instrument
	satellite1 - satellite
	instrument1 - instrument
	satellite2 - satellite
	instrument2 - instrument
	instrument3 - instrument
	instrument4 - instrument
	instrument5 - instrument
	satellite3 - satellite
	instrument6 - instrument
	instrument7 - instrument
	instrument8 - instrument
	instrument9 - instrument
	thermograph0 - mode
	star4 - direction
	star5 - direction
	groundstation1 - direction
	star3 - direction
	star2 - direction
	star0 - direction
	planet6 - direction
)
(:init
	(supports instrument0 thermograph0)
	(calibration_target instrument0 star2)
	(on_board instrument0 satellite0)
	(power_avail satellite0)
	(pointing satellite0 star4)
	(supports instrument1 thermograph0)
	(calibration_target instrument1 star2)
	(calibration_target instrument1 groundstation1)
	(on_board instrument1 satellite1)
	(power_avail satellite1)
	(pointing satellite1 groundstation1)
	(supports instrument2 thermograph0)
	(calibration_target instrument2 star4)
	(calibration_target instrument2 star2)
	(supports instrument3 thermograph0)
	(calibration_target instrument3 star5)
	(calibration_target instrument3 star2)
	(supports instrument4 thermograph0)
	(calibration_target instrument4 groundstation1)
	(calibration_target instrument4 star2)
	(supports instrument5 thermograph0)
	(calibration_target instrument5 star0)
	(calibration_target instrument5 star5)
	(on_board instrument2 satellite2)
	(on_board instrument3 satellite2)
	(on_board instrument4 satellite2)
	(on_board instrument5 satellite2)
	(power_avail satellite2)
	(pointing satellite2 star5)
	(supports instrument6 thermograph0)
	(calibration_target instrument6 star3)
	(supports instrument7 thermograph0)
	(calibration_target instrument7 star3)
	(supports instrument8 thermograph0)
	(calibration_target instrument8 star3)
	(calibration_target instrument8 groundstation1)
	(supports instrument9 thermograph0)
	(calibration_target instrument9 star0)
	(calibration_target instrument9 star2)
	(on_board instrument6 satellite3)
	(on_board instrument7 satellite3)
	(on_board instrument8 satellite3)
	(on_board instrument9 satellite3)
	(power_avail satellite3)
	(pointing satellite3 star4)
)
(:goal (and
	(pointing satellite2 star3)
	(have_image planet6 thermograph0)
))

)
