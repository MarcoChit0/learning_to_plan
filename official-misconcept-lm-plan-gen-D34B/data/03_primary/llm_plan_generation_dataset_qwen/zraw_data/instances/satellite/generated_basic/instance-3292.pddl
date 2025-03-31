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
	instrument7 - instrument
	satellite2 - satellite
	instrument8 - instrument
	instrument9 - instrument
	instrument10 - instrument
	satellite3 - satellite
	instrument11 - instrument
	instrument12 - instrument
	instrument13 - instrument
	image2 - mode
	spectrograph1 - mode
	infrared0 - mode
	star0 - direction
	groundstation2 - direction
	groundstation1 - direction
	planet3 - direction
)
(:init
	(supports instrument0 spectrograph1)
	(supports instrument0 image2)
	(supports instrument0 infrared0)
	(calibration_target instrument0 groundstation2)
	(supports instrument1 infrared0)
	(supports instrument1 spectrograph1)
	(calibration_target instrument1 groundstation2)
	(supports instrument2 spectrograph1)
	(calibration_target instrument2 star0)
	(supports instrument3 infrared0)
	(supports instrument3 image2)
	(supports instrument3 spectrograph1)
	(calibration_target instrument3 groundstation2)
	(on_board instrument0 satellite0)
	(on_board instrument1 satellite0)
	(on_board instrument2 satellite0)
	(on_board instrument3 satellite0)
	(power_avail satellite0)
	(pointing satellite0 planet3)
	(supports instrument4 spectrograph1)
	(calibration_target instrument4 groundstation1)
	(supports instrument5 infrared0)
	(supports instrument5 spectrograph1)
	(supports instrument5 image2)
	(calibration_target instrument5 groundstation1)
	(supports instrument6 spectrograph1)
	(supports instrument6 infrared0)
	(supports instrument6 image2)
	(calibration_target instrument6 groundstation2)
	(supports instrument7 infrared0)
	(supports instrument7 spectrograph1)
	(calibration_target instrument7 groundstation1)
	(on_board instrument4 satellite1)
	(on_board instrument5 satellite1)
	(on_board instrument6 satellite1)
	(on_board instrument7 satellite1)
	(power_avail satellite1)
	(pointing satellite1 groundstation2)
	(supports instrument8 image2)
	(calibration_target instrument8 star0)
	(supports instrument9 spectrograph1)
	(supports instrument9 infrared0)
	(supports instrument9 image2)
	(calibration_target instrument9 groundstation2)
	(supports instrument10 spectrograph1)
	(supports instrument10 image2)
	(calibration_target instrument10 star0)
	(on_board instrument8 satellite2)
	(on_board instrument9 satellite2)
	(on_board instrument10 satellite2)
	(power_avail satellite2)
	(pointing satellite2 planet3)
	(supports instrument11 spectrograph1)
	(supports instrument11 image2)
	(calibration_target instrument11 groundstation1)
	(supports instrument12 image2)
	(calibration_target instrument12 groundstation2)
	(supports instrument13 infrared0)
	(calibration_target instrument13 groundstation1)
	(on_board instrument11 satellite3)
	(on_board instrument12 satellite3)
	(on_board instrument13 satellite3)
	(power_avail satellite3)
	(pointing satellite3 groundstation1)
)
(:goal (and
	(pointing satellite1 groundstation2)
	(pointing satellite2 planet3)
	(have_image planet3 infrared0)
))

)
