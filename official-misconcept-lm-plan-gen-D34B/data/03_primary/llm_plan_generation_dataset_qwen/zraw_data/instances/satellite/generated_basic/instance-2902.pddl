(define (problem strips-sat-x-1)
(:domain satellite)
(:objects
	satellite0 - satellite
	instrument0 - instrument
	satellite1 - satellite
	instrument1 - instrument
	instrument2 - instrument
	instrument3 - instrument
	satellite2 - satellite
	instrument4 - instrument
	instrument5 - instrument
	instrument6 - instrument
	instrument7 - instrument
	satellite3 - satellite
	instrument8 - instrument
	instrument9 - instrument
	instrument10 - instrument
	instrument11 - instrument
	satellite4 - satellite
	instrument12 - instrument
	instrument13 - instrument
	instrument14 - instrument
	instrument15 - instrument
	instrument16 - instrument
	image0 - mode
	infrared2 - mode
	spectrograph1 - mode
	infrared3 - mode
	star0 - direction
	groundstation1 - direction
	planet2 - direction
	planet3 - direction
)
(:init
	(supports instrument0 infrared3)
	(calibration_target instrument0 star0)
	(on_board instrument0 satellite0)
	(power_avail satellite0)
	(pointing satellite0 planet3)
	(supports instrument1 infrared3)
	(supports instrument1 spectrograph1)
	(calibration_target instrument1 groundstation1)
	(supports instrument2 spectrograph1)
	(calibration_target instrument2 groundstation1)
	(supports instrument3 infrared2)
	(supports instrument3 image0)
	(calibration_target instrument3 groundstation1)
	(on_board instrument1 satellite1)
	(on_board instrument2 satellite1)
	(on_board instrument3 satellite1)
	(power_avail satellite1)
	(pointing satellite1 star0)
	(supports instrument4 image0)
	(supports instrument4 infrared2)
	(supports instrument4 spectrograph1)
	(calibration_target instrument4 groundstation1)
	(supports instrument5 image0)
	(calibration_target instrument5 groundstation1)
	(supports instrument6 infrared3)
	(supports instrument6 infrared2)
	(supports instrument6 spectrograph1)
	(calibration_target instrument6 groundstation1)
	(supports instrument7 image0)
	(calibration_target instrument7 groundstation1)
	(on_board instrument4 satellite2)
	(on_board instrument5 satellite2)
	(on_board instrument6 satellite2)
	(on_board instrument7 satellite2)
	(power_avail satellite2)
	(pointing satellite2 planet3)
	(supports instrument8 spectrograph1)
	(supports instrument8 infrared3)
	(calibration_target instrument8 groundstation1)
	(supports instrument9 infrared2)
	(calibration_target instrument9 groundstation1)
	(supports instrument10 spectrograph1)
	(calibration_target instrument10 groundstation1)
	(supports instrument11 infrared2)
	(supports instrument11 spectrograph1)
	(calibration_target instrument11 groundstation1)
	(on_board instrument8 satellite3)
	(on_board instrument9 satellite3)
	(on_board instrument10 satellite3)
	(on_board instrument11 satellite3)
	(power_avail satellite3)
	(pointing satellite3 groundstation1)
	(supports instrument12 infrared2)
	(supports instrument12 image0)
	(supports instrument12 infrared3)
	(calibration_target instrument12 star0)
	(supports instrument13 image0)
	(supports instrument13 spectrograph1)
	(supports instrument13 infrared3)
	(calibration_target instrument13 star0)
	(supports instrument14 spectrograph1)
	(supports instrument14 infrared2)
	(supports instrument14 image0)
	(calibration_target instrument14 star0)
	(supports instrument15 image0)
	(supports instrument15 infrared3)
	(calibration_target instrument15 star0)
	(supports instrument16 infrared2)
	(supports instrument16 image0)
	(calibration_target instrument16 groundstation1)
	(on_board instrument12 satellite4)
	(on_board instrument13 satellite4)
	(on_board instrument14 satellite4)
	(on_board instrument15 satellite4)
	(on_board instrument16 satellite4)
	(power_avail satellite4)
	(pointing satellite4 groundstation1)
)
(:goal (and
	(pointing satellite1 star0)
	(pointing satellite4 star0)
	(have_image planet2 infrared2)
	(have_image planet3 infrared3)
))

)
