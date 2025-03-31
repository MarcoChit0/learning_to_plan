(define (problem strips-sat-x-1)
(:domain satellite)
(:objects
	satellite0 - satellite
	instrument0 - instrument
	instrument1 - instrument
	instrument2 - instrument
	instrument3 - instrument
	satellite1 - satellite
	instrument4 - instrument
	instrument5 - instrument
	instrument6 - instrument
	instrument7 - instrument
	satellite2 - satellite
	instrument8 - instrument
	instrument9 - instrument
	instrument10 - instrument
	satellite3 - satellite
	instrument11 - instrument
	thermograph2 - mode
	infrared1 - mode
	infrared0 - mode
	star2 - direction
	groundstation0 - direction
	groundstation1 - direction
	star3 - direction
	planet4 - direction
)
(:init
	(supports instrument0 infrared1)
	(calibration_target instrument0 groundstation0)
	(supports instrument1 thermograph2)
	(supports instrument1 infrared1)
	(supports instrument1 infrared0)
	(calibration_target instrument1 groundstation0)
	(supports instrument2 infrared0)
	(supports instrument2 thermograph2)
	(supports instrument2 infrared1)
	(calibration_target instrument2 star3)
	(supports instrument3 thermograph2)
	(supports instrument3 infrared0)
	(calibration_target instrument3 star3)
	(on_board instrument0 satellite0)
	(on_board instrument1 satellite0)
	(on_board instrument2 satellite0)
	(on_board instrument3 satellite0)
	(power_avail satellite0)
	(pointing satellite0 star2)
	(supports instrument4 infrared0)
	(supports instrument4 thermograph2)
	(calibration_target instrument4 star2)
	(supports instrument5 thermograph2)
	(supports instrument5 infrared1)
	(supports instrument5 infrared0)
	(calibration_target instrument5 groundstation1)
	(supports instrument6 infrared0)
	(supports instrument6 thermograph2)
	(supports instrument6 infrared1)
	(calibration_target instrument6 groundstation0)
	(supports instrument7 thermograph2)
	(supports instrument7 infrared0)
	(supports instrument7 infrared1)
	(calibration_target instrument7 groundstation1)
	(on_board instrument4 satellite1)
	(on_board instrument5 satellite1)
	(on_board instrument6 satellite1)
	(on_board instrument7 satellite1)
	(power_avail satellite1)
	(pointing satellite1 star3)
	(supports instrument8 infrared1)
	(supports instrument8 infrared0)
	(calibration_target instrument8 groundstation0)
	(supports instrument9 infrared0)
	(supports instrument9 infrared1)
	(supports instrument9 thermograph2)
	(calibration_target instrument9 groundstation0)
	(supports instrument10 infrared0)
	(supports instrument10 thermograph2)
	(calibration_target instrument10 groundstation1)
	(on_board instrument8 satellite2)
	(on_board instrument9 satellite2)
	(on_board instrument10 satellite2)
	(power_avail satellite2)
	(pointing satellite2 groundstation0)
	(supports instrument11 thermograph2)
	(calibration_target instrument11 star3)
	(on_board instrument11 satellite3)
	(power_avail satellite3)
	(pointing satellite3 planet4)
)
(:goal (and
	(pointing satellite0 groundstation0)
	(pointing satellite1 star2)
	(pointing satellite2 groundstation0)
	(pointing satellite3 star2)
	(have_image planet4 infrared0)
))

)
