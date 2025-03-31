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
	instrument8 - instrument
	satellite2 - satellite
	instrument9 - instrument
	instrument10 - instrument
	instrument11 - instrument
	instrument12 - instrument
	instrument13 - instrument
	instrument14 - instrument
	satellite3 - satellite
	instrument15 - instrument
	instrument16 - instrument
	instrument17 - instrument
	instrument18 - instrument
	instrument19 - instrument
	spectrograph1 - mode
	infrared2 - mode
	image0 - mode
	groundstation0 - direction
	phenomenon1 - direction
)
(:init
	(supports instrument0 infrared2)
	(calibration_target instrument0 groundstation0)
	(supports instrument1 infrared2)
	(supports instrument1 image0)
	(supports instrument1 spectrograph1)
	(calibration_target instrument1 groundstation0)
	(supports instrument2 image0)
	(supports instrument2 infrared2)
	(calibration_target instrument2 groundstation0)
	(on_board instrument0 satellite0)
	(on_board instrument1 satellite0)
	(on_board instrument2 satellite0)
	(power_avail satellite0)
	(pointing satellite0 groundstation0)
	(supports instrument3 infrared2)
	(supports instrument3 image0)
	(calibration_target instrument3 groundstation0)
	(supports instrument4 spectrograph1)
	(calibration_target instrument4 groundstation0)
	(supports instrument5 image0)
	(calibration_target instrument5 groundstation0)
	(supports instrument6 infrared2)
	(supports instrument6 spectrograph1)
	(supports instrument6 image0)
	(calibration_target instrument6 groundstation0)
	(supports instrument7 image0)
	(supports instrument7 infrared2)
	(calibration_target instrument7 groundstation0)
	(supports instrument8 infrared2)
	(supports instrument8 spectrograph1)
	(supports instrument8 image0)
	(calibration_target instrument8 groundstation0)
	(on_board instrument3 satellite1)
	(on_board instrument4 satellite1)
	(on_board instrument5 satellite1)
	(on_board instrument6 satellite1)
	(on_board instrument7 satellite1)
	(on_board instrument8 satellite1)
	(power_avail satellite1)
	(pointing satellite1 phenomenon1)
	(supports instrument9 spectrograph1)
	(calibration_target instrument9 groundstation0)
	(supports instrument10 infrared2)
	(supports instrument10 spectrograph1)
	(calibration_target instrument10 groundstation0)
	(supports instrument11 image0)
	(supports instrument11 infrared2)
	(supports instrument11 spectrograph1)
	(calibration_target instrument11 groundstation0)
	(supports instrument12 image0)
	(supports instrument12 infrared2)
	(calibration_target instrument12 groundstation0)
	(supports instrument13 spectrograph1)
	(supports instrument13 infrared2)
	(calibration_target instrument13 groundstation0)
	(supports instrument14 spectrograph1)
	(supports instrument14 image0)
	(calibration_target instrument14 groundstation0)
	(on_board instrument9 satellite2)
	(on_board instrument10 satellite2)
	(on_board instrument11 satellite2)
	(on_board instrument12 satellite2)
	(on_board instrument13 satellite2)
	(on_board instrument14 satellite2)
	(power_avail satellite2)
	(pointing satellite2 phenomenon1)
	(supports instrument15 spectrograph1)
	(calibration_target instrument15 groundstation0)
	(supports instrument16 image0)
	(calibration_target instrument16 groundstation0)
	(supports instrument17 infrared2)
	(supports instrument17 spectrograph1)
	(supports instrument17 image0)
	(calibration_target instrument17 groundstation0)
	(supports instrument18 infrared2)
	(calibration_target instrument18 groundstation0)
	(supports instrument19 spectrograph1)
	(supports instrument19 image0)
	(calibration_target instrument19 groundstation0)
	(on_board instrument15 satellite3)
	(on_board instrument16 satellite3)
	(on_board instrument17 satellite3)
	(on_board instrument18 satellite3)
	(on_board instrument19 satellite3)
	(power_avail satellite3)
	(pointing satellite3 phenomenon1)
)
(:goal (and
	(pointing satellite3 phenomenon1)
	(have_image phenomenon1 spectrograph1)
))

)
