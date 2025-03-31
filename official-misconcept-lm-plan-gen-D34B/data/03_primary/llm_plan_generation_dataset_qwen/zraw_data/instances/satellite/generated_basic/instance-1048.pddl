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
	infrared0 - mode
	spectrograph1 - mode
	thermograph2 - mode
	groundstation3 - direction
	groundstation1 - direction
	groundstation0 - direction
	groundstation2 - direction
	planet4 - direction
	star5 - direction
	star6 - direction
	star7 - direction
	planet8 - direction
)
(:init
	(supports instrument0 thermograph2)
	(calibration_target instrument0 groundstation0)
	(supports instrument1 infrared0)
	(supports instrument1 spectrograph1)
	(supports instrument1 thermograph2)
	(calibration_target instrument1 groundstation3)
	(on_board instrument0 satellite0)
	(on_board instrument1 satellite0)
	(power_avail satellite0)
	(pointing satellite0 star6)
	(supports instrument2 spectrograph1)
	(calibration_target instrument2 groundstation2)
	(supports instrument3 spectrograph1)
	(calibration_target instrument3 groundstation1)
	(supports instrument4 infrared0)
	(supports instrument4 thermograph2)
	(calibration_target instrument4 groundstation0)
	(supports instrument5 infrared0)
	(supports instrument5 thermograph2)
	(supports instrument5 spectrograph1)
	(calibration_target instrument5 groundstation2)
	(on_board instrument2 satellite1)
	(on_board instrument3 satellite1)
	(on_board instrument4 satellite1)
	(on_board instrument5 satellite1)
	(power_avail satellite1)
	(pointing satellite1 groundstation0)
	(supports instrument6 infrared0)
	(supports instrument6 thermograph2)
	(supports instrument6 spectrograph1)
	(calibration_target instrument6 groundstation2)
	(supports instrument7 thermograph2)
	(supports instrument7 spectrograph1)
	(supports instrument7 infrared0)
	(calibration_target instrument7 groundstation2)
	(on_board instrument6 satellite2)
	(on_board instrument7 satellite2)
	(power_avail satellite2)
	(pointing satellite2 groundstation0)
)
(:goal (and
	(pointing satellite0 planet8)
	(pointing satellite2 star7)
	(have_image planet4 spectrograph1)
	(have_image star5 spectrograph1)
	(have_image star6 spectrograph1)
	(have_image star7 thermograph2)
	(have_image planet8 infrared0)
))

)
