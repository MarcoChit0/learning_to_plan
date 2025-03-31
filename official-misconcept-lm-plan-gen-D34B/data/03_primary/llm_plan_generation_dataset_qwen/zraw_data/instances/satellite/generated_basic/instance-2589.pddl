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
	instrument6 - instrument
	instrument7 - instrument
	infrared0 - mode
	spectrograph1 - mode
	thermograph3 - mode
	infrared2 - mode
	star0 - direction
	star1 - direction
	star2 - direction
)
(:init
	(supports instrument0 thermograph3)
	(supports instrument0 infrared0)
	(calibration_target instrument0 star1)
	(supports instrument1 thermograph3)
	(supports instrument1 infrared2)
	(supports instrument1 infrared0)
	(calibration_target instrument1 star1)
	(supports instrument2 infrared0)
	(supports instrument2 thermograph3)
	(calibration_target instrument2 star1)
	(on_board instrument0 satellite0)
	(on_board instrument1 satellite0)
	(on_board instrument2 satellite0)
	(power_avail satellite0)
	(pointing satellite0 star1)
	(supports instrument3 spectrograph1)
	(calibration_target instrument3 star0)
	(supports instrument4 spectrograph1)
	(supports instrument4 infrared0)
	(supports instrument4 thermograph3)
	(calibration_target instrument4 star1)
	(supports instrument5 infrared0)
	(supports instrument5 thermograph3)
	(supports instrument5 infrared2)
	(calibration_target instrument5 star0)
	(supports instrument6 spectrograph1)
	(supports instrument6 infrared0)
	(supports instrument6 infrared2)
	(calibration_target instrument6 star1)
	(supports instrument7 thermograph3)
	(supports instrument7 infrared0)
	(supports instrument7 spectrograph1)
	(calibration_target instrument7 star1)
	(on_board instrument3 satellite1)
	(on_board instrument4 satellite1)
	(on_board instrument5 satellite1)
	(on_board instrument6 satellite1)
	(on_board instrument7 satellite1)
	(power_avail satellite1)
	(pointing satellite1 star0)
)
(:goal (and
	(pointing satellite0 star0)
	(have_image star2 thermograph3)
))

)
