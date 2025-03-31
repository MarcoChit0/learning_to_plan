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
	instrument7 - instrument
	satellite2 - satellite
	instrument8 - instrument
	instrument9 - instrument
	instrument10 - instrument
	instrument11 - instrument
	instrument12 - instrument
	instrument13 - instrument
	infrared0 - mode
	thermograph2 - mode
	spectrograph1 - mode
	groundstation4 - direction
	groundstation1 - direction
	groundstation3 - direction
	star0 - direction
	groundstation2 - direction
	groundstation5 - direction
	planet6 - direction
)
(:init
	(supports instrument0 spectrograph1)
	(supports instrument0 thermograph2)
	(calibration_target instrument0 groundstation4)
	(supports instrument1 thermograph2)
	(calibration_target instrument1 groundstation5)
	(supports instrument2 infrared0)
	(supports instrument2 thermograph2)
	(calibration_target instrument2 groundstation1)
	(calibration_target instrument2 groundstation3)
	(supports instrument3 infrared0)
	(supports instrument3 spectrograph1)
	(calibration_target instrument3 groundstation4)
	(supports instrument4 thermograph2)
	(calibration_target instrument4 groundstation4)
	(calibration_target instrument4 groundstation3)
	(supports instrument5 infrared0)
	(supports instrument5 thermograph2)
	(calibration_target instrument5 groundstation3)
	(on_board instrument0 satellite0)
	(on_board instrument1 satellite0)
	(on_board instrument2 satellite0)
	(on_board instrument3 satellite0)
	(on_board instrument4 satellite0)
	(on_board instrument5 satellite0)
	(power_avail satellite0)
	(pointing satellite0 groundstation4)
	(supports instrument6 spectrograph1)
	(calibration_target instrument6 groundstation1)
	(calibration_target instrument6 groundstation2)
	(supports instrument7 infrared0)
	(supports instrument7 spectrograph1)
	(supports instrument7 thermograph2)
	(calibration_target instrument7 groundstation2)
	(calibration_target instrument7 groundstation3)
	(on_board instrument6 satellite1)
	(on_board instrument7 satellite1)
	(power_avail satellite1)
	(pointing satellite1 star0)
	(supports instrument8 infrared0)
	(calibration_target instrument8 groundstation2)
	(supports instrument9 thermograph2)
	(supports instrument9 infrared0)
	(supports instrument9 spectrograph1)
	(calibration_target instrument9 groundstation5)
	(supports instrument10 infrared0)
	(supports instrument10 thermograph2)
	(calibration_target instrument10 star0)
	(supports instrument11 infrared0)
	(calibration_target instrument11 groundstation5)
	(supports instrument12 spectrograph1)
	(supports instrument12 infrared0)
	(supports instrument12 thermograph2)
	(calibration_target instrument12 groundstation2)
	(calibration_target instrument12 groundstation5)
	(supports instrument13 infrared0)
	(supports instrument13 spectrograph1)
	(supports instrument13 thermograph2)
	(calibration_target instrument13 groundstation5)
	(on_board instrument8 satellite2)
	(on_board instrument9 satellite2)
	(on_board instrument10 satellite2)
	(on_board instrument11 satellite2)
	(on_board instrument12 satellite2)
	(on_board instrument13 satellite2)
	(power_avail satellite2)
	(pointing satellite2 star0)
)
(:goal (and
	(have_image planet6 spectrograph1)
))

)
