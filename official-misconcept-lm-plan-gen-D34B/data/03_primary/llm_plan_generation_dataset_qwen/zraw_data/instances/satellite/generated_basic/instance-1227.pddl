(define (problem strips-sat-x-1)
(:domain satellite)
(:objects
	satellite0 - satellite
	instrument0 - instrument
	instrument1 - instrument
	instrument2 - instrument
	instrument3 - instrument
	instrument4 - instrument
	satellite1 - satellite
	instrument5 - instrument
	instrument6 - instrument
	instrument7 - instrument
	instrument8 - instrument
	thermograph2 - mode
	thermograph1 - mode
	spectrograph3 - mode
	infrared0 - mode
	groundstation0 - direction
	star2 - direction
	star1 - direction
	star3 - direction
	planet4 - direction
	star5 - direction
)
(:init
	(supports instrument0 thermograph1)
	(calibration_target instrument0 star1)
	(supports instrument1 thermograph1)
	(supports instrument1 thermograph2)
	(calibration_target instrument1 star2)
	(supports instrument2 infrared0)
	(supports instrument2 thermograph2)
	(supports instrument2 spectrograph3)
	(calibration_target instrument2 star1)
	(supports instrument3 thermograph2)
	(supports instrument3 thermograph1)
	(calibration_target instrument3 groundstation0)
	(supports instrument4 thermograph1)
	(calibration_target instrument4 star2)
	(on_board instrument0 satellite0)
	(on_board instrument1 satellite0)
	(on_board instrument2 satellite0)
	(on_board instrument3 satellite0)
	(on_board instrument4 satellite0)
	(power_avail satellite0)
	(pointing satellite0 star1)
	(supports instrument5 spectrograph3)
	(supports instrument5 thermograph1)
	(calibration_target instrument5 star2)
	(supports instrument6 infrared0)
	(calibration_target instrument6 star1)
	(supports instrument7 thermograph1)
	(supports instrument7 infrared0)
	(calibration_target instrument7 star2)
	(supports instrument8 thermograph2)
	(supports instrument8 thermograph1)
	(calibration_target instrument8 star1)
	(on_board instrument5 satellite1)
	(on_board instrument6 satellite1)
	(on_board instrument7 satellite1)
	(on_board instrument8 satellite1)
	(power_avail satellite1)
	(pointing satellite1 star3)
)
(:goal (and
	(have_image star3 thermograph2)
	(have_image planet4 spectrograph3)
	(have_image star5 infrared0)
))

)
