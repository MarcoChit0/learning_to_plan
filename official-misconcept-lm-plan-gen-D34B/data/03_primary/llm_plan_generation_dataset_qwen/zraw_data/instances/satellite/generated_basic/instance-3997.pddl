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
	satellite2 - satellite
	instrument7 - instrument
	instrument8 - instrument
	satellite3 - satellite
	instrument9 - instrument
	instrument10 - instrument
	instrument11 - instrument
	satellite4 - satellite
	instrument12 - instrument
	instrument13 - instrument
	instrument14 - instrument
	instrument15 - instrument
	instrument16 - instrument
	thermograph2 - mode
	thermograph0 - mode
	spectrograph1 - mode
	infrared3 - mode
	groundstation3 - direction
	groundstation0 - direction
	star1 - direction
	star2 - direction
	star4 - direction
)
(:init
	(supports instrument0 thermograph0)
	(supports instrument0 infrared3)
	(calibration_target instrument0 star2)
	(supports instrument1 thermograph2)
	(supports instrument1 thermograph0)
	(calibration_target instrument1 star2)
	(supports instrument2 thermograph0)
	(supports instrument2 thermograph2)
	(supports instrument2 spectrograph1)
	(calibration_target instrument2 groundstation0)
	(supports instrument3 spectrograph1)
	(supports instrument3 thermograph2)
	(calibration_target instrument3 groundstation3)
	(on_board instrument0 satellite0)
	(on_board instrument1 satellite0)
	(on_board instrument2 satellite0)
	(on_board instrument3 satellite0)
	(power_avail satellite0)
	(pointing satellite0 star2)
	(supports instrument4 spectrograph1)
	(supports instrument4 thermograph0)
	(supports instrument4 thermograph2)
	(calibration_target instrument4 star1)
	(supports instrument5 infrared3)
	(supports instrument5 spectrograph1)
	(calibration_target instrument5 star2)
	(supports instrument6 thermograph2)
	(supports instrument6 infrared3)
	(supports instrument6 thermograph0)
	(calibration_target instrument6 groundstation3)
	(on_board instrument4 satellite1)
	(on_board instrument5 satellite1)
	(on_board instrument6 satellite1)
	(power_avail satellite1)
	(pointing satellite1 groundstation0)
	(supports instrument7 infrared3)
	(calibration_target instrument7 star1)
	(supports instrument8 spectrograph1)
	(calibration_target instrument8 star1)
	(on_board instrument7 satellite2)
	(on_board instrument8 satellite2)
	(power_avail satellite2)
	(pointing satellite2 star4)
	(supports instrument9 thermograph2)
	(supports instrument9 thermograph0)
	(calibration_target instrument9 star1)
	(supports instrument10 thermograph0)
	(supports instrument10 thermograph2)
	(supports instrument10 spectrograph1)
	(calibration_target instrument10 star1)
	(supports instrument11 thermograph0)
	(supports instrument11 thermograph2)
	(supports instrument11 infrared3)
	(calibration_target instrument11 groundstation3)
	(on_board instrument9 satellite3)
	(on_board instrument10 satellite3)
	(on_board instrument11 satellite3)
	(power_avail satellite3)
	(pointing satellite3 groundstation3)
	(supports instrument12 infrared3)
	(supports instrument12 spectrograph1)
	(supports instrument12 thermograph2)
	(calibration_target instrument12 star1)
	(supports instrument13 thermograph2)
	(supports instrument13 infrared3)
	(calibration_target instrument13 groundstation0)
	(supports instrument14 infrared3)
	(calibration_target instrument14 star1)
	(supports instrument15 thermograph2)
	(supports instrument15 infrared3)
	(calibration_target instrument15 star2)
	(supports instrument16 thermograph2)
	(calibration_target instrument16 star2)
	(on_board instrument12 satellite4)
	(on_board instrument13 satellite4)
	(on_board instrument14 satellite4)
	(on_board instrument15 satellite4)
	(on_board instrument16 satellite4)
	(power_avail satellite4)
	(pointing satellite4 star1)
)
(:goal (and
	(pointing satellite1 groundstation3)
	(pointing satellite2 star4)
	(have_image star4 thermograph0)
))

)
