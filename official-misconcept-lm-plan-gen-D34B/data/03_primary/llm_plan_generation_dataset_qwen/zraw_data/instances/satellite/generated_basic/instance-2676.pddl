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
	spectrograph0 - mode
	thermograph2 - mode
	thermograph1 - mode
	spectrograph3 - mode
	groundstation0 - direction
	star2 - direction
	groundstation3 - direction
	groundstation1 - direction
	planet4 - direction
)
(:init
	(supports instrument0 spectrograph0)
	(supports instrument0 thermograph2)
	(supports instrument0 thermograph1)
	(calibration_target instrument0 groundstation0)
	(supports instrument1 spectrograph0)
	(calibration_target instrument1 star2)
	(supports instrument2 thermograph1)
	(supports instrument2 spectrograph0)
	(supports instrument2 spectrograph3)
	(calibration_target instrument2 groundstation0)
	(supports instrument3 thermograph1)
	(supports instrument3 spectrograph3)
	(supports instrument3 thermograph2)
	(calibration_target instrument3 star2)
	(supports instrument4 spectrograph0)
	(supports instrument4 spectrograph3)
	(supports instrument4 thermograph2)
	(calibration_target instrument4 groundstation3)
	(on_board instrument0 satellite0)
	(on_board instrument1 satellite0)
	(on_board instrument2 satellite0)
	(on_board instrument3 satellite0)
	(on_board instrument4 satellite0)
	(power_avail satellite0)
	(pointing satellite0 planet4)
	(supports instrument5 spectrograph0)
	(supports instrument5 spectrograph3)
	(supports instrument5 thermograph2)
	(calibration_target instrument5 groundstation3)
	(supports instrument6 spectrograph3)
	(supports instrument6 thermograph2)
	(supports instrument6 spectrograph0)
	(calibration_target instrument6 groundstation1)
	(supports instrument7 thermograph2)
	(calibration_target instrument7 groundstation1)
	(on_board instrument5 satellite1)
	(on_board instrument6 satellite1)
	(on_board instrument7 satellite1)
	(power_avail satellite1)
	(pointing satellite1 groundstation3)
)
(:goal (and
	(have_image planet4 spectrograph0)
))

)
