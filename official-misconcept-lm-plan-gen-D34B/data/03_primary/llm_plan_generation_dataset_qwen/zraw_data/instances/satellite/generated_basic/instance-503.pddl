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
	satellite3 - satellite
	instrument12 - instrument
	instrument13 - instrument
	instrument14 - instrument
	instrument15 - instrument
	instrument16 - instrument
	spectrograph2 - mode
	infrared3 - mode
	infrared1 - mode
	infrared0 - mode
	star2 - direction
	star1 - direction
	groundstation3 - direction
	star4 - direction
	star0 - direction
	planet5 - direction
	phenomenon6 - direction
	phenomenon7 - direction
)
(:init
	(supports instrument0 infrared1)
	(supports instrument0 spectrograph2)
	(supports instrument0 infrared3)
	(calibration_target instrument0 star2)
	(supports instrument1 spectrograph2)
	(supports instrument1 infrared3)
	(supports instrument1 infrared1)
	(calibration_target instrument1 star2)
	(supports instrument2 infrared0)
	(supports instrument2 infrared1)
	(calibration_target instrument2 star2)
	(supports instrument3 infrared0)
	(supports instrument3 infrared3)
	(calibration_target instrument3 star1)
	(on_board instrument0 satellite0)
	(on_board instrument1 satellite0)
	(on_board instrument2 satellite0)
	(on_board instrument3 satellite0)
	(power_avail satellite0)
	(pointing satellite0 star1)
	(supports instrument4 spectrograph2)
	(calibration_target instrument4 star4)
	(supports instrument5 infrared1)
	(supports instrument5 spectrograph2)
	(calibration_target instrument5 star4)
	(supports instrument6 infrared0)
	(calibration_target instrument6 groundstation3)
	(on_board instrument4 satellite1)
	(on_board instrument5 satellite1)
	(on_board instrument6 satellite1)
	(power_avail satellite1)
	(pointing satellite1 planet5)
	(supports instrument7 infrared1)
	(calibration_target instrument7 star4)
	(supports instrument8 spectrograph2)
	(supports instrument8 infrared0)
	(supports instrument8 infrared1)
	(calibration_target instrument8 star0)
	(supports instrument9 infrared0)
	(supports instrument9 infrared1)
	(supports instrument9 spectrograph2)
	(calibration_target instrument9 groundstation3)
	(supports instrument10 infrared0)
	(calibration_target instrument10 star4)
	(supports instrument11 infrared0)
	(supports instrument11 infrared1)
	(supports instrument11 infrared3)
	(calibration_target instrument11 groundstation3)
	(on_board instrument7 satellite2)
	(on_board instrument8 satellite2)
	(on_board instrument9 satellite2)
	(on_board instrument10 satellite2)
	(on_board instrument11 satellite2)
	(power_avail satellite2)
	(pointing satellite2 star2)
	(supports instrument12 infrared3)
	(supports instrument12 spectrograph2)
	(supports instrument12 infrared0)
	(calibration_target instrument12 groundstation3)
	(supports instrument13 infrared0)
	(supports instrument13 infrared3)
	(calibration_target instrument13 star4)
	(supports instrument14 infrared0)
	(supports instrument14 spectrograph2)
	(supports instrument14 infrared3)
	(calibration_target instrument14 star0)
	(supports instrument15 infrared0)
	(supports instrument15 infrared3)
	(supports instrument15 spectrograph2)
	(calibration_target instrument15 star0)
	(supports instrument16 spectrograph2)
	(calibration_target instrument16 star0)
	(on_board instrument12 satellite3)
	(on_board instrument13 satellite3)
	(on_board instrument14 satellite3)
	(on_board instrument15 satellite3)
	(on_board instrument16 satellite3)
	(power_avail satellite3)
	(pointing satellite3 phenomenon6)
)
(:goal (and
	(have_image planet5 infrared0)
	(have_image phenomenon6 infrared3)
	(have_image phenomenon7 infrared3)
))

)
