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
	satellite2 - satellite
	instrument5 - instrument
	instrument6 - instrument
	instrument7 - instrument
	instrument8 - instrument
	satellite3 - satellite
	instrument9 - instrument
	thermograph0 - mode
	star2 - direction
	groundstation5 - direction
	star1 - direction
	star0 - direction
	groundstation3 - direction
	groundstation4 - direction
	star6 - direction
)
(:init
	(supports instrument0 thermograph0)
	(calibration_target instrument0 groundstation4)
	(calibration_target instrument0 groundstation3)
	(supports instrument1 thermograph0)
	(calibration_target instrument1 star0)
	(supports instrument2 thermograph0)
	(calibration_target instrument2 star0)
	(on_board instrument0 satellite0)
	(on_board instrument1 satellite0)
	(on_board instrument2 satellite0)
	(power_avail satellite0)
	(pointing satellite0 star2)
	(supports instrument3 thermograph0)
	(calibration_target instrument3 groundstation4)
	(calibration_target instrument3 star0)
	(supports instrument4 thermograph0)
	(calibration_target instrument4 groundstation4)
	(on_board instrument3 satellite1)
	(on_board instrument4 satellite1)
	(power_avail satellite1)
	(pointing satellite1 groundstation5)
	(supports instrument5 thermograph0)
	(calibration_target instrument5 star0)
	(supports instrument6 thermograph0)
	(calibration_target instrument6 star0)
	(supports instrument7 thermograph0)
	(calibration_target instrument7 star0)
	(calibration_target instrument7 star1)
	(supports instrument8 thermograph0)
	(calibration_target instrument8 groundstation3)
	(on_board instrument5 satellite2)
	(on_board instrument6 satellite2)
	(on_board instrument7 satellite2)
	(on_board instrument8 satellite2)
	(power_avail satellite2)
	(pointing satellite2 groundstation3)
	(supports instrument9 thermograph0)
	(calibration_target instrument9 groundstation4)
	(calibration_target instrument9 groundstation3)
	(on_board instrument9 satellite3)
	(power_avail satellite3)
	(pointing satellite3 star6)
)
(:goal (and
	(pointing satellite3 groundstation4)
	(have_image star6 thermograph0)
))

)
