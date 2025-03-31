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
	satellite3 - satellite
	instrument8 - instrument
	instrument9 - instrument
	instrument10 - instrument
	instrument11 - instrument
	instrument12 - instrument
	instrument13 - instrument
	satellite4 - satellite
	instrument14 - instrument
	instrument15 - instrument
	instrument16 - instrument
	instrument17 - instrument
	infrared3 - mode
	thermograph2 - mode
	thermograph1 - mode
	spectrograph0 - mode
	star3 - direction
	groundstation2 - direction
	groundstation1 - direction
	groundstation0 - direction
	star4 - direction
	phenomenon5 - direction
)
(:init
	(supports instrument0 thermograph2)
	(calibration_target instrument0 groundstation0)
	(supports instrument1 spectrograph0)
	(calibration_target instrument1 groundstation0)
	(supports instrument2 infrared3)
	(supports instrument2 spectrograph0)
	(calibration_target instrument2 groundstation2)
	(on_board instrument0 satellite0)
	(on_board instrument1 satellite0)
	(on_board instrument2 satellite0)
	(power_avail satellite0)
	(pointing satellite0 star3)
	(supports instrument3 infrared3)
	(calibration_target instrument3 star3)
	(supports instrument4 spectrograph0)
	(calibration_target instrument4 groundstation2)
	(on_board instrument3 satellite1)
	(on_board instrument4 satellite1)
	(power_avail satellite1)
	(pointing satellite1 groundstation2)
	(supports instrument5 infrared3)
	(calibration_target instrument5 groundstation1)
	(supports instrument6 thermograph2)
	(supports instrument6 thermograph1)
	(supports instrument6 infrared3)
	(calibration_target instrument6 groundstation1)
	(supports instrument7 spectrograph0)
	(supports instrument7 thermograph1)
	(supports instrument7 thermograph2)
	(calibration_target instrument7 groundstation2)
	(on_board instrument5 satellite2)
	(on_board instrument6 satellite2)
	(on_board instrument7 satellite2)
	(power_avail satellite2)
	(pointing satellite2 groundstation2)
	(supports instrument8 infrared3)
	(calibration_target instrument8 groundstation0)
	(supports instrument9 spectrograph0)
	(supports instrument9 thermograph2)
	(supports instrument9 infrared3)
	(calibration_target instrument9 groundstation0)
	(supports instrument10 spectrograph0)
	(supports instrument10 infrared3)
	(calibration_target instrument10 star3)
	(supports instrument11 thermograph2)
	(supports instrument11 infrared3)
	(calibration_target instrument11 groundstation1)
	(supports instrument12 spectrograph0)
	(calibration_target instrument12 groundstation0)
	(supports instrument13 thermograph1)
	(calibration_target instrument13 groundstation2)
	(on_board instrument8 satellite3)
	(on_board instrument9 satellite3)
	(on_board instrument10 satellite3)
	(on_board instrument11 satellite3)
	(on_board instrument12 satellite3)
	(on_board instrument13 satellite3)
	(power_avail satellite3)
	(pointing satellite3 star3)
	(supports instrument14 infrared3)
	(supports instrument14 thermograph2)
	(calibration_target instrument14 groundstation0)
	(supports instrument15 spectrograph0)
	(supports instrument15 thermograph1)
	(calibration_target instrument15 groundstation1)
	(supports instrument16 thermograph2)
	(calibration_target instrument16 groundstation0)
	(supports instrument17 thermograph2)
	(supports instrument17 spectrograph0)
	(supports instrument17 thermograph1)
	(calibration_target instrument17 groundstation0)
	(on_board instrument14 satellite4)
	(on_board instrument15 satellite4)
	(on_board instrument16 satellite4)
	(on_board instrument17 satellite4)
	(power_avail satellite4)
	(pointing satellite4 groundstation0)
)
(:goal (and
	(pointing satellite0 groundstation1)
	(pointing satellite1 phenomenon5)
	(have_image star4 thermograph1)
	(have_image phenomenon5 thermograph2)
))

)
