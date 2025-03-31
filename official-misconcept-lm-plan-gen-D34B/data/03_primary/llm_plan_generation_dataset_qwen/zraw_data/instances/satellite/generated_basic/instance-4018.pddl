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
	satellite2 - satellite
	instrument6 - instrument
	instrument7 - instrument
	instrument8 - instrument
	instrument9 - instrument
	satellite3 - satellite
	instrument10 - instrument
	instrument11 - instrument
	instrument12 - instrument
	instrument13 - instrument
	instrument14 - instrument
	instrument15 - instrument
	satellite4 - satellite
	instrument16 - instrument
	instrument17 - instrument
	instrument18 - instrument
	instrument19 - instrument
	instrument20 - instrument
	satellite5 - satellite
	instrument21 - instrument
	instrument22 - instrument
	instrument23 - instrument
	instrument24 - instrument
	instrument25 - instrument
	spectrograph2 - mode
	spectrograph1 - mode
	infrared0 - mode
	groundstation1 - direction
	star0 - direction
	star2 - direction
	planet3 - direction
)
(:init
	(supports instrument0 infrared0)
	(calibration_target instrument0 star0)
	(supports instrument1 infrared0)
	(supports instrument1 spectrograph1)
	(calibration_target instrument1 star0)
	(supports instrument2 infrared0)
	(supports instrument2 spectrograph1)
	(calibration_target instrument2 star2)
	(on_board instrument0 satellite0)
	(on_board instrument1 satellite0)
	(on_board instrument2 satellite0)
	(power_avail satellite0)
	(pointing satellite0 groundstation1)
	(supports instrument3 infrared0)
	(calibration_target instrument3 groundstation1)
	(supports instrument4 spectrograph1)
	(calibration_target instrument4 star2)
	(supports instrument5 infrared0)
	(calibration_target instrument5 star0)
	(on_board instrument3 satellite1)
	(on_board instrument4 satellite1)
	(on_board instrument5 satellite1)
	(power_avail satellite1)
	(pointing satellite1 star0)
	(supports instrument6 spectrograph1)
	(calibration_target instrument6 star0)
	(supports instrument7 spectrograph2)
	(supports instrument7 spectrograph1)
	(supports instrument7 infrared0)
	(calibration_target instrument7 star2)
	(supports instrument8 infrared0)
	(supports instrument8 spectrograph1)
	(calibration_target instrument8 star2)
	(supports instrument9 spectrograph2)
	(calibration_target instrument9 star0)
	(on_board instrument6 satellite2)
	(on_board instrument7 satellite2)
	(on_board instrument8 satellite2)
	(on_board instrument9 satellite2)
	(power_avail satellite2)
	(pointing satellite2 star0)
	(supports instrument10 spectrograph2)
	(calibration_target instrument10 groundstation1)
	(supports instrument11 spectrograph1)
	(supports instrument11 spectrograph2)
	(supports instrument11 infrared0)
	(calibration_target instrument11 star2)
	(supports instrument12 spectrograph1)
	(supports instrument12 spectrograph2)
	(calibration_target instrument12 star0)
	(supports instrument13 infrared0)
	(calibration_target instrument13 groundstation1)
	(supports instrument14 spectrograph2)
	(supports instrument14 spectrograph1)
	(calibration_target instrument14 star0)
	(supports instrument15 spectrograph1)
	(supports instrument15 infrared0)
	(supports instrument15 spectrograph2)
	(calibration_target instrument15 star0)
	(on_board instrument10 satellite3)
	(on_board instrument11 satellite3)
	(on_board instrument12 satellite3)
	(on_board instrument13 satellite3)
	(on_board instrument14 satellite3)
	(on_board instrument15 satellite3)
	(power_avail satellite3)
	(pointing satellite3 star0)
	(supports instrument16 infrared0)
	(calibration_target instrument16 star0)
	(supports instrument17 infrared0)
	(supports instrument17 spectrograph2)
	(supports instrument17 spectrograph1)
	(calibration_target instrument17 star0)
	(supports instrument18 spectrograph1)
	(calibration_target instrument18 star0)
	(supports instrument19 spectrograph2)
	(calibration_target instrument19 star0)
	(supports instrument20 infrared0)
	(supports instrument20 spectrograph2)
	(calibration_target instrument20 groundstation1)
	(on_board instrument16 satellite4)
	(on_board instrument17 satellite4)
	(on_board instrument18 satellite4)
	(on_board instrument19 satellite4)
	(on_board instrument20 satellite4)
	(power_avail satellite4)
	(pointing satellite4 star0)
	(supports instrument21 spectrograph1)
	(supports instrument21 infrared0)
	(supports instrument21 spectrograph2)
	(calibration_target instrument21 star0)
	(supports instrument22 spectrograph1)
	(calibration_target instrument22 groundstation1)
	(supports instrument23 infrared0)
	(supports instrument23 spectrograph1)
	(supports instrument23 spectrograph2)
	(calibration_target instrument23 groundstation1)
	(supports instrument24 spectrograph1)
	(calibration_target instrument24 star0)
	(supports instrument25 infrared0)
	(calibration_target instrument25 star2)
	(on_board instrument21 satellite5)
	(on_board instrument22 satellite5)
	(on_board instrument23 satellite5)
	(on_board instrument24 satellite5)
	(on_board instrument25 satellite5)
	(power_avail satellite5)
	(pointing satellite5 planet3)
)
(:goal (and
	(have_image planet3 spectrograph1)
))

)
