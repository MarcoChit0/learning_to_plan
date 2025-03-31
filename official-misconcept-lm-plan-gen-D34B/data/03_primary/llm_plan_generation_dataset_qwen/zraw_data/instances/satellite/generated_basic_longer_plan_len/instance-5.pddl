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
	spectrograph0 - mode
	infrared3 - mode
	infrared2 - mode
	spectrograph1 - mode
	groundstation0 - direction
	groundstation4 - direction
	groundstation1 - direction
	star2 - direction
	star3 - direction
	phenomenon5 - direction
	star6 - direction
	star7 - direction
	planet8 - direction
	star9 - direction
	star10 - direction
)
(:init
	(supports instrument0 infrared2)
	(supports instrument0 infrared3)
	(calibration_target instrument0 groundstation4)
	(supports instrument1 spectrograph0)
	(supports instrument1 infrared3)
	(supports instrument1 infrared2)
	(calibration_target instrument1 groundstation1)
	(supports instrument2 infrared3)
	(supports instrument2 infrared2)
	(supports instrument2 spectrograph0)
	(calibration_target instrument2 groundstation1)
	(supports instrument3 infrared2)
	(supports instrument3 spectrograph1)
	(supports instrument3 infrared3)
	(calibration_target instrument3 star2)
	(supports instrument4 infrared3)
	(supports instrument4 spectrograph0)
	(supports instrument4 infrared2)
	(calibration_target instrument4 groundstation1)
	(on_board instrument0 satellite0)
	(on_board instrument1 satellite0)
	(on_board instrument2 satellite0)
	(on_board instrument3 satellite0)
	(on_board instrument4 satellite0)
	(power_avail satellite0)
	(pointing satellite0 groundstation0)
	(supports instrument5 infrared3)
	(supports instrument5 infrared2)
	(calibration_target instrument5 groundstation4)
	(supports instrument6 spectrograph1)
	(supports instrument6 infrared3)
	(calibration_target instrument6 star3)
	(on_board instrument5 satellite1)
	(on_board instrument6 satellite1)
	(power_avail satellite1)
	(pointing satellite1 planet8)
	(supports instrument7 infrared2)
	(supports instrument7 infrared3)
	(calibration_target instrument7 star3)
	(supports instrument8 spectrograph0)
	(supports instrument8 infrared2)
	(supports instrument8 infrared3)
	(calibration_target instrument8 groundstation0)
	(supports instrument9 infrared3)
	(supports instrument9 spectrograph1)
	(supports instrument9 infrared2)
	(calibration_target instrument9 groundstation4)
	(supports instrument10 infrared2)
	(calibration_target instrument10 groundstation1)
	(on_board instrument7 satellite2)
	(on_board instrument8 satellite2)
	(on_board instrument9 satellite2)
	(on_board instrument10 satellite2)
	(power_avail satellite2)
	(pointing satellite2 groundstation4)
	(supports instrument11 infrared3)
	(calibration_target instrument11 star3)
	(supports instrument12 spectrograph1)
	(supports instrument12 infrared3)
	(supports instrument12 infrared2)
	(calibration_target instrument12 groundstation1)
	(supports instrument13 spectrograph0)
	(calibration_target instrument13 star3)
	(supports instrument14 infrared3)
	(calibration_target instrument14 groundstation4)
	(on_board instrument11 satellite3)
	(on_board instrument12 satellite3)
	(on_board instrument13 satellite3)
	(on_board instrument14 satellite3)
	(power_avail satellite3)
	(pointing satellite3 phenomenon5)
	(supports instrument15 infrared3)
	(calibration_target instrument15 groundstation4)
	(supports instrument16 infrared3)
	(calibration_target instrument16 groundstation1)
	(supports instrument17 infrared3)
	(supports instrument17 infrared2)
	(calibration_target instrument17 star2)
	(supports instrument18 infrared3)
	(supports instrument18 infrared2)
	(calibration_target instrument18 star3)
	(on_board instrument15 satellite4)
	(on_board instrument16 satellite4)
	(on_board instrument17 satellite4)
	(on_board instrument18 satellite4)
	(power_avail satellite4)
	(pointing satellite4 star10)
)
(:goal (and
	(pointing satellite0 star6)
	(pointing satellite2 star6)
	(have_image phenomenon5 infrared3)
	(have_image star6 infrared2)
	(have_image star7 infrared3)
	(have_image planet8 spectrograph0)
	(have_image star9 infrared2)
	(have_image star10 infrared3)
))

)
