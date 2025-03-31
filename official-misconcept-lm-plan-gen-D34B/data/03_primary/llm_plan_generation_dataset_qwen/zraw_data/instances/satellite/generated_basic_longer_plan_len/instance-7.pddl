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
	instrument13 - instrument
	instrument14 - instrument
	satellite4 - satellite
	instrument15 - instrument
	instrument16 - instrument
	thermograph3 - mode
	thermograph1 - mode
	spectrograph2 - mode
	infrared0 - mode
	star2 - direction
	star4 - direction
	star1 - direction
	groundstation3 - direction
	groundstation0 - direction
	planet5 - direction
	star6 - direction
	star7 - direction
	phenomenon8 - direction
	planet9 - direction
	star10 - direction
)
(:init
	(supports instrument0 thermograph3)
	(supports instrument0 thermograph1)
	(calibration_target instrument0 groundstation3)
	(supports instrument1 spectrograph2)
	(supports instrument1 thermograph1)
	(supports instrument1 infrared0)
	(calibration_target instrument1 star2)
	(supports instrument2 thermograph1)
	(supports instrument2 thermograph3)
	(supports instrument2 infrared0)
	(calibration_target instrument2 star4)
	(on_board instrument0 satellite0)
	(on_board instrument1 satellite0)
	(on_board instrument2 satellite0)
	(power_avail satellite0)
	(pointing satellite0 groundstation0)
	(supports instrument3 thermograph3)
	(supports instrument3 thermograph1)
	(supports instrument3 spectrograph2)
	(calibration_target instrument3 star1)
	(supports instrument4 thermograph3)
	(supports instrument4 infrared0)
	(calibration_target instrument4 groundstation0)
	(supports instrument5 infrared0)
	(supports instrument5 thermograph1)
	(calibration_target instrument5 star1)
	(supports instrument6 thermograph1)
	(supports instrument6 thermograph3)
	(calibration_target instrument6 star4)
	(supports instrument7 spectrograph2)
	(supports instrument7 infrared0)
	(supports instrument7 thermograph3)
	(calibration_target instrument7 star4)
	(on_board instrument3 satellite1)
	(on_board instrument4 satellite1)
	(on_board instrument5 satellite1)
	(on_board instrument6 satellite1)
	(on_board instrument7 satellite1)
	(power_avail satellite1)
	(pointing satellite1 star10)
	(supports instrument8 infrared0)
	(calibration_target instrument8 groundstation3)
	(supports instrument9 infrared0)
	(supports instrument9 spectrograph2)
	(supports instrument9 thermograph1)
	(calibration_target instrument9 star4)
	(supports instrument10 thermograph1)
	(calibration_target instrument10 star1)
	(supports instrument11 thermograph1)
	(calibration_target instrument11 groundstation0)
	(on_board instrument8 satellite2)
	(on_board instrument9 satellite2)
	(on_board instrument10 satellite2)
	(on_board instrument11 satellite2)
	(power_avail satellite2)
	(pointing satellite2 planet9)
	(supports instrument12 infrared0)
	(supports instrument12 spectrograph2)
	(supports instrument12 thermograph1)
	(calibration_target instrument12 star1)
	(supports instrument13 thermograph1)
	(supports instrument13 infrared0)
	(calibration_target instrument13 star1)
	(supports instrument14 thermograph3)
	(calibration_target instrument14 groundstation3)
	(on_board instrument12 satellite3)
	(on_board instrument13 satellite3)
	(on_board instrument14 satellite3)
	(power_avail satellite3)
	(pointing satellite3 star6)
	(supports instrument15 thermograph3)
	(calibration_target instrument15 groundstation0)
	(supports instrument16 infrared0)
	(supports instrument16 spectrograph2)
	(calibration_target instrument16 groundstation0)
	(on_board instrument15 satellite4)
	(on_board instrument16 satellite4)
	(power_avail satellite4)
	(pointing satellite4 phenomenon8)
)
(:goal (and
	(pointing satellite1 star7)
	(pointing satellite2 groundstation3)
	(pointing satellite3 phenomenon8)
	(have_image planet5 thermograph1)
	(have_image star6 thermograph3)
	(have_image star7 thermograph3)
	(have_image phenomenon8 thermograph1)
	(have_image planet9 infrared0)
	(have_image star10 thermograph1)
))

)
