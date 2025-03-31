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
	satellite3 - satellite
	instrument8 - instrument
	instrument9 - instrument
	instrument10 - instrument
	thermograph0 - mode
	groundstation4 - direction
	star2 - direction
	groundstation5 - direction
	groundstation0 - direction
	groundstation3 - direction
	groundstation1 - direction
	star6 - direction
)
(:init
	(supports instrument0 thermograph0)
	(calibration_target instrument0 groundstation0)
	(supports instrument1 thermograph0)
	(calibration_target instrument1 groundstation1)
	(calibration_target instrument1 groundstation3)
	(supports instrument2 thermograph0)
	(calibration_target instrument2 groundstation5)
	(calibration_target instrument2 groundstation1)
	(on_board instrument0 satellite0)
	(on_board instrument1 satellite0)
	(on_board instrument2 satellite0)
	(power_avail satellite0)
	(pointing satellite0 groundstation5)
	(supports instrument3 thermograph0)
	(calibration_target instrument3 star2)
	(calibration_target instrument3 groundstation4)
	(supports instrument4 thermograph0)
	(calibration_target instrument4 groundstation1)
	(calibration_target instrument4 groundstation4)
	(supports instrument5 thermograph0)
	(calibration_target instrument5 groundstation0)
	(calibration_target instrument5 star2)
	(supports instrument6 thermograph0)
	(calibration_target instrument6 groundstation3)
	(on_board instrument3 satellite1)
	(on_board instrument4 satellite1)
	(on_board instrument5 satellite1)
	(on_board instrument6 satellite1)
	(power_avail satellite1)
	(pointing satellite1 star2)
	(supports instrument7 thermograph0)
	(calibration_target instrument7 groundstation3)
	(on_board instrument7 satellite2)
	(power_avail satellite2)
	(pointing satellite2 groundstation3)
	(supports instrument8 thermograph0)
	(calibration_target instrument8 groundstation1)
	(supports instrument9 thermograph0)
	(calibration_target instrument9 groundstation0)
	(calibration_target instrument9 groundstation5)
	(supports instrument10 thermograph0)
	(calibration_target instrument10 groundstation1)
	(calibration_target instrument10 groundstation3)
	(on_board instrument8 satellite3)
	(on_board instrument9 satellite3)
	(on_board instrument10 satellite3)
	(power_avail satellite3)
	(pointing satellite3 groundstation0)
)
(:goal (and
	(have_image star6 thermograph0)
))

)
