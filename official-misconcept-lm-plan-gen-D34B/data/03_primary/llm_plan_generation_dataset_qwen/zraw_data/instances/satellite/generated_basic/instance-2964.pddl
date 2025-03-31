(define (problem strips-sat-x-1)
(:domain satellite)
(:objects
	satellite0 - satellite
	instrument0 - instrument
	instrument1 - instrument
	satellite1 - satellite
	instrument2 - instrument
	instrument3 - instrument
	instrument4 - instrument
	instrument5 - instrument
	instrument6 - instrument
	satellite2 - satellite
	instrument7 - instrument
	instrument8 - instrument
	instrument9 - instrument
	instrument10 - instrument
	satellite3 - satellite
	instrument11 - instrument
	instrument12 - instrument
	instrument13 - instrument
	instrument14 - instrument
	satellite4 - satellite
	instrument15 - instrument
	instrument16 - instrument
	instrument17 - instrument
	instrument18 - instrument
	spectrograph1 - mode
	infrared3 - mode
	infrared0 - mode
	spectrograph2 - mode
	star0 - direction
	star2 - direction
	groundstation3 - direction
	groundstation1 - direction
	planet4 - direction
	star5 - direction
)
(:init
	(supports instrument0 infrared3)
	(supports instrument0 infrared0)
	(supports instrument0 spectrograph2)
	(calibration_target instrument0 groundstation3)
	(supports instrument1 infrared0)
	(supports instrument1 spectrograph1)
	(supports instrument1 infrared3)
	(calibration_target instrument1 star0)
	(on_board instrument0 satellite0)
	(on_board instrument1 satellite0)
	(power_avail satellite0)
	(pointing satellite0 planet4)
	(supports instrument2 infrared0)
	(calibration_target instrument2 star0)
	(supports instrument3 infrared3)
	(supports instrument3 spectrograph1)
	(calibration_target instrument3 groundstation3)
	(supports instrument4 infrared3)
	(supports instrument4 spectrograph2)
	(supports instrument4 spectrograph1)
	(calibration_target instrument4 star2)
	(supports instrument5 spectrograph1)
	(supports instrument5 infrared0)
	(calibration_target instrument5 groundstation3)
	(supports instrument6 spectrograph2)
	(calibration_target instrument6 groundstation3)
	(on_board instrument2 satellite1)
	(on_board instrument3 satellite1)
	(on_board instrument4 satellite1)
	(on_board instrument5 satellite1)
	(on_board instrument6 satellite1)
	(power_avail satellite1)
	(pointing satellite1 groundstation3)
	(supports instrument7 spectrograph1)
	(supports instrument7 infrared3)
	(supports instrument7 infrared0)
	(calibration_target instrument7 star0)
	(supports instrument8 infrared0)
	(calibration_target instrument8 groundstation1)
	(supports instrument9 spectrograph1)
	(calibration_target instrument9 star2)
	(supports instrument10 infrared3)
	(supports instrument10 infrared0)
	(supports instrument10 spectrograph1)
	(calibration_target instrument10 groundstation1)
	(on_board instrument7 satellite2)
	(on_board instrument8 satellite2)
	(on_board instrument9 satellite2)
	(on_board instrument10 satellite2)
	(power_avail satellite2)
	(pointing satellite2 star5)
	(supports instrument11 infrared0)
	(supports instrument11 spectrograph1)
	(calibration_target instrument11 groundstation3)
	(supports instrument12 spectrograph2)
	(supports instrument12 spectrograph1)
	(calibration_target instrument12 star2)
	(supports instrument13 spectrograph2)
	(supports instrument13 infrared3)
	(calibration_target instrument13 star0)
	(supports instrument14 spectrograph2)
	(supports instrument14 infrared3)
	(supports instrument14 infrared0)
	(calibration_target instrument14 groundstation1)
	(on_board instrument11 satellite3)
	(on_board instrument12 satellite3)
	(on_board instrument13 satellite3)
	(on_board instrument14 satellite3)
	(power_avail satellite3)
	(pointing satellite3 groundstation3)
	(supports instrument15 spectrograph1)
	(supports instrument15 spectrograph2)
	(supports instrument15 infrared3)
	(calibration_target instrument15 star2)
	(supports instrument16 spectrograph2)
	(supports instrument16 infrared3)
	(calibration_target instrument16 star2)
	(supports instrument17 spectrograph1)
	(calibration_target instrument17 groundstation3)
	(supports instrument18 spectrograph1)
	(calibration_target instrument18 groundstation1)
	(on_board instrument15 satellite4)
	(on_board instrument16 satellite4)
	(on_board instrument17 satellite4)
	(on_board instrument18 satellite4)
	(power_avail satellite4)
	(pointing satellite4 groundstation3)
)
(:goal (and
	(pointing satellite1 star5)
	(pointing satellite2 star2)
	(pointing satellite4 planet4)
	(have_image planet4 infrared3)
	(have_image star5 infrared3)
))

)
