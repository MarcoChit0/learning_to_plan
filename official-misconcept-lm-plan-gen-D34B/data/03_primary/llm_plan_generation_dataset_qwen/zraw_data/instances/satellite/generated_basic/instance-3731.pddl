(define (problem strips-sat-x-1)
(:domain satellite)
(:objects
	satellite0 - satellite
	instrument0 - instrument
	instrument1 - instrument
	instrument2 - instrument
	instrument3 - instrument
	instrument4 - instrument
	satellite1 - satellite
	instrument5 - instrument
	satellite2 - satellite
	instrument6 - instrument
	instrument7 - instrument
	instrument8 - instrument
	instrument9 - instrument
	instrument10 - instrument
	instrument11 - instrument
	image0 - mode
	infrared1 - mode
	thermograph2 - mode
	groundstation1 - direction
	groundstation2 - direction
	star4 - direction
	star5 - direction
	groundstation3 - direction
	groundstation0 - direction
	star6 - direction
)
(:init
	(supports instrument0 image0)
	(calibration_target instrument0 star5)
	(supports instrument1 image0)
	(supports instrument1 infrared1)
	(supports instrument1 thermograph2)
	(calibration_target instrument1 groundstation0)
	(calibration_target instrument1 star5)
	(supports instrument2 infrared1)
	(supports instrument2 thermograph2)
	(supports instrument2 image0)
	(calibration_target instrument2 groundstation1)
	(calibration_target instrument2 groundstation0)
	(supports instrument3 image0)
	(supports instrument3 infrared1)
	(calibration_target instrument3 groundstation3)
	(calibration_target instrument3 star5)
	(supports instrument4 image0)
	(supports instrument4 thermograph2)
	(calibration_target instrument4 star4)
	(calibration_target instrument4 groundstation2)
	(on_board instrument0 satellite0)
	(on_board instrument1 satellite0)
	(on_board instrument2 satellite0)
	(on_board instrument3 satellite0)
	(on_board instrument4 satellite0)
	(power_avail satellite0)
	(pointing satellite0 groundstation0)
	(supports instrument5 thermograph2)
	(supports instrument5 infrared1)
	(supports instrument5 image0)
	(calibration_target instrument5 star4)
	(calibration_target instrument5 star5)
	(on_board instrument5 satellite1)
	(power_avail satellite1)
	(pointing satellite1 groundstation0)
	(supports instrument6 thermograph2)
	(supports instrument6 image0)
	(calibration_target instrument6 groundstation0)
	(supports instrument7 thermograph2)
	(supports instrument7 image0)
	(calibration_target instrument7 star5)
	(calibration_target instrument7 star4)
	(supports instrument8 image0)
	(supports instrument8 infrared1)
	(supports instrument8 thermograph2)
	(calibration_target instrument8 star4)
	(calibration_target instrument8 groundstation0)
	(supports instrument9 thermograph2)
	(calibration_target instrument9 star5)
	(supports instrument10 image0)
	(calibration_target instrument10 groundstation0)
	(calibration_target instrument10 groundstation3)
	(supports instrument11 thermograph2)
	(calibration_target instrument11 groundstation0)
	(on_board instrument6 satellite2)
	(on_board instrument7 satellite2)
	(on_board instrument8 satellite2)
	(on_board instrument9 satellite2)
	(on_board instrument10 satellite2)
	(on_board instrument11 satellite2)
	(power_avail satellite2)
	(pointing satellite2 groundstation1)
)
(:goal (and
	(pointing satellite1 groundstation1)
	(have_image star6 thermograph2)
))

)
