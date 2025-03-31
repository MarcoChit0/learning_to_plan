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
	spectrograph3 - mode
	thermograph0 - mode
	thermograph2 - mode
	spectrograph1 - mode
	groundstation2 - direction
	groundstation1 - direction
	star0 - direction
	star3 - direction
	phenomenon4 - direction
	star5 - direction
)
(:init
	(supports instrument0 thermograph2)
	(supports instrument0 thermograph0)
	(supports instrument0 spectrograph3)
	(calibration_target instrument0 star0)
	(supports instrument1 thermograph2)
	(supports instrument1 spectrograph3)
	(calibration_target instrument1 star0)
	(supports instrument2 spectrograph1)
	(supports instrument2 spectrograph3)
	(calibration_target instrument2 groundstation1)
	(supports instrument3 thermograph0)
	(supports instrument3 thermograph2)
	(supports instrument3 spectrograph1)
	(calibration_target instrument3 groundstation1)
	(on_board instrument0 satellite0)
	(on_board instrument1 satellite0)
	(on_board instrument2 satellite0)
	(on_board instrument3 satellite0)
	(power_avail satellite0)
	(pointing satellite0 star3)
	(supports instrument4 spectrograph3)
	(supports instrument4 spectrograph1)
	(calibration_target instrument4 star0)
	(supports instrument5 thermograph2)
	(supports instrument5 spectrograph3)
	(supports instrument5 spectrograph1)
	(calibration_target instrument5 groundstation2)
	(supports instrument6 spectrograph3)
	(supports instrument6 thermograph0)
	(supports instrument6 spectrograph1)
	(calibration_target instrument6 groundstation1)
	(supports instrument7 thermograph0)
	(calibration_target instrument7 star0)
	(on_board instrument4 satellite1)
	(on_board instrument5 satellite1)
	(on_board instrument6 satellite1)
	(on_board instrument7 satellite1)
	(power_avail satellite1)
	(pointing satellite1 phenomenon4)
)
(:goal (and
	(pointing satellite0 groundstation2)
	(pointing satellite1 groundstation2)
	(have_image star3 spectrograph3)
	(have_image phenomenon4 thermograph2)
	(have_image star5 spectrograph1)
))

)
