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
	instrument9 - instrument
	instrument10 - instrument
	instrument11 - instrument
	instrument12 - instrument
	satellite3 - satellite
	instrument13 - instrument
	instrument14 - instrument
	instrument15 - instrument
	instrument16 - instrument
	instrument17 - instrument
	infrared3 - mode
	infrared0 - mode
	infrared1 - mode
	spectrograph2 - mode
	groundstation4 - direction
	groundstation0 - direction
	star3 - direction
	groundstation2 - direction
	groundstation1 - direction
	planet5 - direction
	phenomenon6 - direction
	planet7 - direction
)
(:init
	(supports instrument0 infrared1)
	(calibration_target instrument0 star3)
	(supports instrument1 infrared0)
	(supports instrument1 infrared1)
	(calibration_target instrument1 groundstation1)
	(supports instrument2 infrared3)
	(supports instrument2 infrared0)
	(supports instrument2 spectrograph2)
	(calibration_target instrument2 groundstation1)
	(supports instrument3 spectrograph2)
	(supports instrument3 infrared0)
	(calibration_target instrument3 groundstation0)
	(on_board instrument0 satellite0)
	(on_board instrument1 satellite0)
	(on_board instrument2 satellite0)
	(on_board instrument3 satellite0)
	(power_avail satellite0)
	(pointing satellite0 groundstation4)
	(supports instrument4 infrared0)
	(calibration_target instrument4 groundstation0)
	(supports instrument5 spectrograph2)
	(supports instrument5 infrared3)
	(calibration_target instrument5 groundstation0)
	(supports instrument6 infrared1)
	(supports instrument6 spectrograph2)
	(calibration_target instrument6 groundstation0)
	(on_board instrument4 satellite1)
	(on_board instrument5 satellite1)
	(on_board instrument6 satellite1)
	(power_avail satellite1)
	(pointing satellite1 groundstation2)
	(supports instrument7 spectrograph2)
	(calibration_target instrument7 groundstation0)
	(supports instrument8 infrared3)
	(supports instrument8 infrared0)
	(supports instrument8 infrared1)
	(calibration_target instrument8 groundstation2)
	(supports instrument9 infrared0)
	(supports instrument9 infrared3)
	(calibration_target instrument9 groundstation1)
	(supports instrument10 infrared3)
	(supports instrument10 infrared0)
	(supports instrument10 spectrograph2)
	(calibration_target instrument10 groundstation1)
	(supports instrument11 spectrograph2)
	(calibration_target instrument11 groundstation1)
	(supports instrument12 infrared1)
	(calibration_target instrument12 star3)
	(on_board instrument7 satellite2)
	(on_board instrument8 satellite2)
	(on_board instrument9 satellite2)
	(on_board instrument10 satellite2)
	(on_board instrument11 satellite2)
	(on_board instrument12 satellite2)
	(power_avail satellite2)
	(pointing satellite2 phenomenon6)
	(supports instrument13 infrared0)
	(supports instrument13 spectrograph2)
	(supports instrument13 infrared1)
	(calibration_target instrument13 star3)
	(supports instrument14 spectrograph2)
	(calibration_target instrument14 groundstation1)
	(supports instrument15 spectrograph2)
	(supports instrument15 infrared3)
	(supports instrument15 infrared1)
	(calibration_target instrument15 groundstation1)
	(supports instrument16 spectrograph2)
	(calibration_target instrument16 groundstation2)
	(supports instrument17 infrared3)
	(calibration_target instrument17 groundstation1)
	(on_board instrument13 satellite3)
	(on_board instrument14 satellite3)
	(on_board instrument15 satellite3)
	(on_board instrument16 satellite3)
	(on_board instrument17 satellite3)
	(power_avail satellite3)
	(pointing satellite3 groundstation0)
)
(:goal (and
	(pointing satellite0 groundstation1)
	(pointing satellite1 groundstation1)
	(have_image planet5 infrared1)
	(have_image phenomenon6 infrared0)
	(have_image planet7 infrared1)
))

)
