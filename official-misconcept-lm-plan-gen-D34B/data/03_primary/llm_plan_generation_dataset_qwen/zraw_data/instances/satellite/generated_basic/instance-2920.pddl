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
	instrument6 - instrument
	instrument7 - instrument
	instrument8 - instrument
	instrument9 - instrument
	satellite2 - satellite
	instrument10 - instrument
	instrument11 - instrument
	instrument12 - instrument
	instrument13 - instrument
	satellite3 - satellite
	instrument14 - instrument
	instrument15 - instrument
	instrument16 - instrument
	instrument17 - instrument
	instrument18 - instrument
	instrument19 - instrument
	satellite4 - satellite
	instrument20 - instrument
	spectrograph3 - mode
	infrared0 - mode
	infrared2 - mode
	spectrograph1 - mode
	groundstation0 - direction
	star1 - direction
	phenomenon2 - direction
	star3 - direction
)
(:init
	(supports instrument0 spectrograph3)
	(calibration_target instrument0 groundstation0)
	(supports instrument1 infrared0)
	(supports instrument1 spectrograph3)
	(calibration_target instrument1 star1)
	(supports instrument2 spectrograph3)
	(calibration_target instrument2 groundstation0)
	(supports instrument3 infrared2)
	(supports instrument3 infrared0)
	(calibration_target instrument3 groundstation0)
	(supports instrument4 infrared0)
	(supports instrument4 spectrograph3)
	(calibration_target instrument4 groundstation0)
	(on_board instrument0 satellite0)
	(on_board instrument1 satellite0)
	(on_board instrument2 satellite0)
	(on_board instrument3 satellite0)
	(on_board instrument4 satellite0)
	(power_avail satellite0)
	(pointing satellite0 star3)
	(supports instrument5 infrared2)
	(supports instrument5 infrared0)
	(calibration_target instrument5 groundstation0)
	(supports instrument6 spectrograph1)
	(supports instrument6 infrared2)
	(supports instrument6 infrared0)
	(calibration_target instrument6 groundstation0)
	(supports instrument7 infrared0)
	(supports instrument7 spectrograph3)
	(supports instrument7 spectrograph1)
	(calibration_target instrument7 groundstation0)
	(supports instrument8 spectrograph3)
	(supports instrument8 infrared0)
	(supports instrument8 spectrograph1)
	(calibration_target instrument8 star1)
	(supports instrument9 spectrograph1)
	(supports instrument9 infrared2)
	(supports instrument9 spectrograph3)
	(calibration_target instrument9 groundstation0)
	(on_board instrument5 satellite1)
	(on_board instrument6 satellite1)
	(on_board instrument7 satellite1)
	(on_board instrument8 satellite1)
	(on_board instrument9 satellite1)
	(power_avail satellite1)
	(pointing satellite1 star1)
	(supports instrument10 infrared0)
	(supports instrument10 infrared2)
	(calibration_target instrument10 star1)
	(supports instrument11 infrared2)
	(supports instrument11 spectrograph1)
	(supports instrument11 spectrograph3)
	(calibration_target instrument11 groundstation0)
	(supports instrument12 spectrograph3)
	(calibration_target instrument12 star1)
	(supports instrument13 spectrograph1)
	(supports instrument13 infrared2)
	(calibration_target instrument13 star1)
	(on_board instrument10 satellite2)
	(on_board instrument11 satellite2)
	(on_board instrument12 satellite2)
	(on_board instrument13 satellite2)
	(power_avail satellite2)
	(pointing satellite2 star1)
	(supports instrument14 infrared2)
	(supports instrument14 spectrograph3)
	(calibration_target instrument14 groundstation0)
	(supports instrument15 infrared2)
	(supports instrument15 spectrograph3)
	(supports instrument15 spectrograph1)
	(calibration_target instrument15 star1)
	(supports instrument16 spectrograph1)
	(calibration_target instrument16 star1)
	(supports instrument17 spectrograph1)
	(supports instrument17 infrared2)
	(calibration_target instrument17 star1)
	(supports instrument18 infrared2)
	(calibration_target instrument18 groundstation0)
	(supports instrument19 spectrograph3)
	(supports instrument19 infrared0)
	(supports instrument19 infrared2)
	(calibration_target instrument19 star1)
	(on_board instrument14 satellite3)
	(on_board instrument15 satellite3)
	(on_board instrument16 satellite3)
	(on_board instrument17 satellite3)
	(on_board instrument18 satellite3)
	(on_board instrument19 satellite3)
	(power_avail satellite3)
	(pointing satellite3 star1)
	(supports instrument20 spectrograph3)
	(supports instrument20 infrared2)
	(calibration_target instrument20 star1)
	(on_board instrument20 satellite4)
	(power_avail satellite4)
	(pointing satellite4 star1)
)
(:goal (and
	(pointing satellite0 star1)
	(pointing satellite1 groundstation0)
	(pointing satellite2 groundstation0)
	(pointing satellite3 groundstation0)
	(have_image phenomenon2 spectrograph3)
	(have_image star3 infrared2)
))

)
