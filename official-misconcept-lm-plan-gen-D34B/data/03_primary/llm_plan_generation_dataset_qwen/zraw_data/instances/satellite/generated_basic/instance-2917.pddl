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
	satellite2 - satellite
	instrument6 - instrument
	instrument7 - instrument
	instrument8 - instrument
	instrument9 - instrument
	satellite3 - satellite
	instrument10 - instrument
	instrument11 - instrument
	satellite4 - satellite
	instrument12 - instrument
	instrument13 - instrument
	instrument14 - instrument
	instrument15 - instrument
	image2 - mode
	infrared0 - mode
	infrared1 - mode
	spectrograph3 - mode
	star0 - direction
	groundstation1 - direction
	star2 - direction
	star3 - direction
)
(:init
	(supports instrument0 spectrograph3)
	(supports instrument0 infrared0)
	(calibration_target instrument0 star0)
	(supports instrument1 infrared0)
	(supports instrument1 image2)
	(supports instrument1 infrared1)
	(calibration_target instrument1 star0)
	(on_board instrument0 satellite0)
	(on_board instrument1 satellite0)
	(power_avail satellite0)
	(pointing satellite0 star0)
	(supports instrument2 image2)
	(supports instrument2 spectrograph3)
	(supports instrument2 infrared0)
	(calibration_target instrument2 groundstation1)
	(supports instrument3 infrared0)
	(supports instrument3 spectrograph3)
	(supports instrument3 infrared1)
	(calibration_target instrument3 star0)
	(supports instrument4 infrared1)
	(calibration_target instrument4 star0)
	(supports instrument5 infrared1)
	(calibration_target instrument5 star0)
	(on_board instrument2 satellite1)
	(on_board instrument3 satellite1)
	(on_board instrument4 satellite1)
	(on_board instrument5 satellite1)
	(power_avail satellite1)
	(pointing satellite1 star0)
	(supports instrument6 image2)
	(supports instrument6 infrared1)
	(calibration_target instrument6 star0)
	(supports instrument7 image2)
	(supports instrument7 infrared1)
	(supports instrument7 spectrograph3)
	(calibration_target instrument7 star0)
	(supports instrument8 infrared1)
	(calibration_target instrument8 star0)
	(supports instrument9 image2)
	(calibration_target instrument9 star0)
	(on_board instrument6 satellite2)
	(on_board instrument7 satellite2)
	(on_board instrument8 satellite2)
	(on_board instrument9 satellite2)
	(power_avail satellite2)
	(pointing satellite2 star0)
	(supports instrument10 image2)
	(supports instrument10 spectrograph3)
	(calibration_target instrument10 groundstation1)
	(supports instrument11 infrared0)
	(supports instrument11 infrared1)
	(supports instrument11 spectrograph3)
	(calibration_target instrument11 star0)
	(on_board instrument10 satellite3)
	(on_board instrument11 satellite3)
	(power_avail satellite3)
	(pointing satellite3 star2)
	(supports instrument12 spectrograph3)
	(supports instrument12 infrared1)
	(supports instrument12 image2)
	(calibration_target instrument12 star0)
	(supports instrument13 infrared1)
	(supports instrument13 image2)
	(calibration_target instrument13 groundstation1)
	(supports instrument14 spectrograph3)
	(supports instrument14 infrared1)
	(calibration_target instrument14 star0)
	(supports instrument15 infrared1)
	(calibration_target instrument15 groundstation1)
	(on_board instrument12 satellite4)
	(on_board instrument13 satellite4)
	(on_board instrument14 satellite4)
	(on_board instrument15 satellite4)
	(power_avail satellite4)
	(pointing satellite4 star3)
)
(:goal (and
	(pointing satellite0 star2)
	(pointing satellite1 star0)
	(pointing satellite4 star3)
	(have_image star2 image2)
	(have_image star3 spectrograph3)
))

)
