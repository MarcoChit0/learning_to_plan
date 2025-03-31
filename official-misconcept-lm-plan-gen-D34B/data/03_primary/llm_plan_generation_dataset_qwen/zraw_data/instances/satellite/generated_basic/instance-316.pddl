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
	satellite2 - satellite
	instrument6 - instrument
	instrument7 - instrument
	image3 - mode
	infrared0 - mode
	thermograph2 - mode
	spectrograph1 - mode
	groundstation5 - direction
	groundstation4 - direction
	groundstation3 - direction
	groundstation2 - direction
	star0 - direction
	groundstation1 - direction
	star6 - direction
)
(:init
	(supports instrument0 spectrograph1)
	(calibration_target instrument0 groundstation2)
	(calibration_target instrument0 groundstation4)
	(supports instrument1 spectrograph1)
	(supports instrument1 image3)
	(calibration_target instrument1 star0)
	(supports instrument2 thermograph2)
	(supports instrument2 spectrograph1)
	(supports instrument2 infrared0)
	(calibration_target instrument2 groundstation4)
	(calibration_target instrument2 groundstation5)
	(on_board instrument0 satellite0)
	(on_board instrument1 satellite0)
	(on_board instrument2 satellite0)
	(power_avail satellite0)
	(pointing satellite0 groundstation3)
	(supports instrument3 spectrograph1)
	(calibration_target instrument3 groundstation1)
	(calibration_target instrument3 groundstation3)
	(supports instrument4 spectrograph1)
	(supports instrument4 infrared0)
	(supports instrument4 thermograph2)
	(calibration_target instrument4 groundstation3)
	(calibration_target instrument4 groundstation4)
	(supports instrument5 spectrograph1)
	(calibration_target instrument5 groundstation3)
	(calibration_target instrument5 star0)
	(on_board instrument3 satellite1)
	(on_board instrument4 satellite1)
	(on_board instrument5 satellite1)
	(power_avail satellite1)
	(pointing satellite1 groundstation3)
	(supports instrument6 image3)
	(calibration_target instrument6 groundstation2)
	(supports instrument7 infrared0)
	(calibration_target instrument7 groundstation1)
	(calibration_target instrument7 star0)
	(on_board instrument6 satellite2)
	(on_board instrument7 satellite2)
	(power_avail satellite2)
	(pointing satellite2 groundstation3)
)
(:goal (and
	(pointing satellite2 groundstation3)
	(have_image star6 spectrograph1)
))

)
