(define (problem strips-sat-x-1)
(:domain satellite)
(:objects
	satellite0 - satellite
	instrument0 - instrument
	instrument1 - instrument
	instrument2 - instrument
	instrument3 - instrument
	instrument4 - instrument
	instrument5 - instrument
	satellite1 - satellite
	instrument6 - instrument
	satellite2 - satellite
	instrument7 - instrument
	instrument8 - instrument
	instrument9 - instrument
	instrument10 - instrument
	instrument11 - instrument
	thermograph2 - mode
	image3 - mode
	thermograph0 - mode
	infrared1 - mode
	groundstation1 - direction
	groundstation2 - direction
	groundstation0 - direction
	planet3 - direction
	star4 - direction
)
(:init
	(supports instrument0 image3)
	(supports instrument0 infrared1)
	(supports instrument0 thermograph0)
	(calibration_target instrument0 groundstation2)
	(supports instrument1 image3)
	(supports instrument1 thermograph2)
	(calibration_target instrument1 groundstation2)
	(supports instrument2 image3)
	(supports instrument2 thermograph0)
	(calibration_target instrument2 groundstation0)
	(supports instrument3 thermograph0)
	(supports instrument3 thermograph2)
	(supports instrument3 image3)
	(calibration_target instrument3 groundstation2)
	(supports instrument4 image3)
	(supports instrument4 thermograph2)
	(calibration_target instrument4 groundstation0)
	(supports instrument5 infrared1)
	(supports instrument5 image3)
	(calibration_target instrument5 groundstation1)
	(on_board instrument0 satellite0)
	(on_board instrument1 satellite0)
	(on_board instrument2 satellite0)
	(on_board instrument3 satellite0)
	(on_board instrument4 satellite0)
	(on_board instrument5 satellite0)
	(power_avail satellite0)
	(pointing satellite0 groundstation1)
	(supports instrument6 thermograph0)
	(supports instrument6 infrared1)
	(supports instrument6 image3)
	(calibration_target instrument6 groundstation0)
	(on_board instrument6 satellite1)
	(power_avail satellite1)
	(pointing satellite1 groundstation1)
	(supports instrument7 thermograph0)
	(supports instrument7 thermograph2)
	(supports instrument7 image3)
	(calibration_target instrument7 groundstation1)
	(supports instrument8 image3)
	(calibration_target instrument8 groundstation0)
	(supports instrument9 infrared1)
	(calibration_target instrument9 groundstation2)
	(supports instrument10 thermograph0)
	(supports instrument10 infrared1)
	(supports instrument10 image3)
	(calibration_target instrument10 groundstation0)
	(supports instrument11 image3)
	(supports instrument11 thermograph0)
	(calibration_target instrument11 groundstation0)
	(on_board instrument7 satellite2)
	(on_board instrument8 satellite2)
	(on_board instrument9 satellite2)
	(on_board instrument10 satellite2)
	(on_board instrument11 satellite2)
	(power_avail satellite2)
	(pointing satellite2 planet3)
)
(:goal (and
	(pointing satellite0 star4)
	(pointing satellite2 groundstation2)
	(have_image planet3 thermograph0)
	(have_image star4 infrared1)
))

)
