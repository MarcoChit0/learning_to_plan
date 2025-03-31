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
	spectrograph2 - mode
	thermograph1 - mode
	infrared0 - mode
	star0 - direction
	star1 - direction
	planet2 - direction
	star3 - direction
	planet4 - direction
	phenomenon5 - direction
)
(:init
	(supports instrument0 spectrograph2)
	(calibration_target instrument0 star1)
	(supports instrument1 infrared0)
	(supports instrument1 spectrograph2)
	(calibration_target instrument1 star1)
	(supports instrument2 infrared0)
	(supports instrument2 thermograph1)
	(calibration_target instrument2 star0)
	(supports instrument3 infrared0)
	(supports instrument3 spectrograph2)
	(supports instrument3 thermograph1)
	(calibration_target instrument3 star0)
	(supports instrument4 spectrograph2)
	(supports instrument4 thermograph1)
	(supports instrument4 infrared0)
	(calibration_target instrument4 star0)
	(supports instrument5 infrared0)
	(calibration_target instrument5 star0)
	(on_board instrument0 satellite0)
	(on_board instrument1 satellite0)
	(on_board instrument2 satellite0)
	(on_board instrument3 satellite0)
	(on_board instrument4 satellite0)
	(on_board instrument5 satellite0)
	(power_avail satellite0)
	(pointing satellite0 star1)
	(supports instrument6 spectrograph2)
	(supports instrument6 thermograph1)
	(calibration_target instrument6 star1)
	(supports instrument7 infrared0)
	(supports instrument7 spectrograph2)
	(supports instrument7 thermograph1)
	(calibration_target instrument7 star1)
	(on_board instrument6 satellite1)
	(on_board instrument7 satellite1)
	(power_avail satellite1)
	(pointing satellite1 phenomenon5)
)
(:goal (and
	(pointing satellite0 star3)
	(have_image planet2 infrared0)
	(have_image star3 spectrograph2)
	(have_image planet4 thermograph1)
	(have_image phenomenon5 thermograph1)
))

)
