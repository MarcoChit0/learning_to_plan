(define (problem strips-sat-x-1)
(:domain satellite)
(:objects
	satellite0 - satellite
	instrument0 - instrument
	instrument1 - instrument
	instrument2 - instrument
	instrument3 - instrument
	instrument4 - instrument
	instrument5 - instrument
	satellite1 - satellite
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
	infrared0 - mode
	spectrograph2 - mode
	infrared1 - mode
	star0 - direction
	groundstation1 - direction
	groundstation3 - direction
	star4 - direction
	groundstation5 - direction
	groundstation2 - direction
	phenomenon6 - direction
	phenomenon7 - direction
	phenomenon8 - direction
	planet9 - direction
)
(:init
	(supports instrument0 infrared0)
	(supports instrument0 infrared1)
	(supports instrument0 spectrograph2)
	(calibration_target instrument0 groundstation1)
	(supports instrument1 spectrograph2)
	(calibration_target instrument1 groundstation3)
	(supports instrument2 infrared1)
	(calibration_target instrument2 groundstation3)
	(calibration_target instrument2 groundstation5)
	(supports instrument3 spectrograph2)
	(supports instrument3 infrared1)
	(supports instrument3 infrared0)
	(calibration_target instrument3 star0)
	(calibration_target instrument3 groundstation5)
	(supports instrument4 infrared1)
	(calibration_target instrument4 star0)
	(supports instrument5 infrared1)
	(supports instrument5 spectrograph2)
	(calibration_target instrument5 groundstation1)
	(calibration_target instrument5 star0)
	(on_board instrument0 satellite0)
	(on_board instrument1 satellite0)
	(on_board instrument2 satellite0)
	(on_board instrument3 satellite0)
	(on_board instrument4 satellite0)
	(on_board instrument5 satellite0)
	(power_avail satellite0)
	(pointing satellite0 groundstation1)
	(supports instrument6 spectrograph2)
	(supports instrument6 infrared1)
	(calibration_target instrument6 groundstation3)
	(on_board instrument6 satellite1)
	(power_avail satellite1)
	(pointing satellite1 star4)
	(supports instrument7 spectrograph2)
	(supports instrument7 infrared1)
	(supports instrument7 infrared0)
	(calibration_target instrument7 groundstation1)
	(calibration_target instrument7 groundstation3)
	(supports instrument8 spectrograph2)
	(supports instrument8 infrared1)
	(supports instrument8 infrared0)
	(calibration_target instrument8 groundstation2)
	(calibration_target instrument8 groundstation1)
	(supports instrument9 infrared0)
	(supports instrument9 spectrograph2)
	(calibration_target instrument9 star0)
	(calibration_target instrument9 groundstation5)
	(supports instrument10 infrared0)
	(calibration_target instrument10 groundstation3)
	(calibration_target instrument10 groundstation1)
	(on_board instrument7 satellite2)
	(on_board instrument8 satellite2)
	(on_board instrument9 satellite2)
	(on_board instrument10 satellite2)
	(power_avail satellite2)
	(pointing satellite2 groundstation5)
	(supports instrument11 spectrograph2)
	(supports instrument11 infrared1)
	(calibration_target instrument11 groundstation5)
	(supports instrument12 spectrograph2)
	(supports instrument12 infrared1)
	(calibration_target instrument12 groundstation5)
	(supports instrument13 infrared1)
	(supports instrument13 spectrograph2)
	(supports instrument13 infrared0)
	(calibration_target instrument13 star4)
	(supports instrument14 infrared0)
	(supports instrument14 infrared1)
	(supports instrument14 spectrograph2)
	(calibration_target instrument14 groundstation2)
	(calibration_target instrument14 groundstation5)
	(on_board instrument11 satellite3)
	(on_board instrument12 satellite3)
	(on_board instrument13 satellite3)
	(on_board instrument14 satellite3)
	(power_avail satellite3)
	(pointing satellite3 star0)
)
(:goal (and
	(pointing satellite0 star0)
	(have_image phenomenon6 infrared0)
	(have_image phenomenon7 infrared0)
	(have_image phenomenon8 infrared0)
	(have_image planet9 infrared0)
))

)
