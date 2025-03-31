(define (problem strips-sat-x-1)
(:domain satellite)
(:objects
	satellite0 - satellite
	instrument0 - instrument
	instrument1 - instrument
	instrument2 - instrument
	satellite1 - satellite
	instrument3 - instrument
	satellite2 - satellite
	instrument4 - instrument
	instrument5 - instrument
	instrument6 - instrument
	satellite3 - satellite
	instrument7 - instrument
	instrument8 - instrument
	instrument9 - instrument
	thermograph0 - mode
	groundstation4 - direction
	star2 - direction
	groundstation5 - direction
	groundstation0 - direction
	groundstation1 - direction
	star3 - direction
	phenomenon6 - direction
)
(:init
	(supports instrument0 thermograph0)
	(calibration_target instrument0 groundstation1)
	(calibration_target instrument0 star2)
	(supports instrument1 thermograph0)
	(calibration_target instrument1 star2)
	(supports instrument2 thermograph0)
	(calibration_target instrument2 groundstation4)
	(on_board instrument0 satellite0)
	(on_board instrument1 satellite0)
	(on_board instrument2 satellite0)
	(power_avail satellite0)
	(pointing satellite0 phenomenon6)
	(supports instrument3 thermograph0)
	(calibration_target instrument3 groundstation1)
	(calibration_target instrument3 groundstation0)
	(on_board instrument3 satellite1)
	(power_avail satellite1)
	(pointing satellite1 groundstation1)
	(supports instrument4 thermograph0)
	(calibration_target instrument4 groundstation5)
	(calibration_target instrument4 star3)
	(supports instrument5 thermograph0)
	(calibration_target instrument5 star3)
	(calibration_target instrument5 star2)
	(supports instrument6 thermograph0)
	(calibration_target instrument6 groundstation5)
	(on_board instrument4 satellite2)
	(on_board instrument5 satellite2)
	(on_board instrument6 satellite2)
	(power_avail satellite2)
	(pointing satellite2 groundstation5)
	(supports instrument7 thermograph0)
	(calibration_target instrument7 groundstation1)
	(supports instrument8 thermograph0)
	(calibration_target instrument8 groundstation1)
	(calibration_target instrument8 groundstation0)
	(supports instrument9 thermograph0)
	(calibration_target instrument9 star3)
	(on_board instrument7 satellite3)
	(on_board instrument8 satellite3)
	(on_board instrument9 satellite3)
	(power_avail satellite3)
	(pointing satellite3 phenomenon6)
)
(:goal (and
	(have_image phenomenon6 thermograph0)
))

)
