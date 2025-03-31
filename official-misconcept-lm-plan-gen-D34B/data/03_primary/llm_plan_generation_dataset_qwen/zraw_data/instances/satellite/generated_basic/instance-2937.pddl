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
	instrument9 - instrument
	thermograph1 - mode
	spectrograph2 - mode
	spectrograph0 - mode
	groundstation2 - direction
	star0 - direction
	groundstation4 - direction
	star3 - direction
	groundstation1 - direction
	planet5 - direction
	phenomenon6 - direction
	star7 - direction
	star8 - direction
)
(:init
	(supports instrument0 spectrograph2)
	(calibration_target instrument0 groundstation4)
	(supports instrument1 spectrograph0)
	(supports instrument1 thermograph1)
	(supports instrument1 spectrograph2)
	(calibration_target instrument1 star0)
	(supports instrument2 spectrograph0)
	(supports instrument2 thermograph1)
	(supports instrument2 spectrograph2)
	(calibration_target instrument2 star0)
	(supports instrument3 spectrograph2)
	(supports instrument3 thermograph1)
	(calibration_target instrument3 groundstation1)
	(supports instrument4 thermograph1)
	(supports instrument4 spectrograph0)
	(calibration_target instrument4 groundstation4)
	(on_board instrument0 satellite0)
	(on_board instrument1 satellite0)
	(on_board instrument2 satellite0)
	(on_board instrument3 satellite0)
	(on_board instrument4 satellite0)
	(power_avail satellite0)
	(pointing satellite0 star7)
	(supports instrument5 spectrograph2)
	(calibration_target instrument5 star3)
	(supports instrument6 spectrograph0)
	(supports instrument6 thermograph1)
	(supports instrument6 spectrograph2)
	(calibration_target instrument6 star3)
	(supports instrument7 spectrograph2)
	(calibration_target instrument7 star3)
	(supports instrument8 spectrograph0)
	(supports instrument8 spectrograph2)
	(calibration_target instrument8 groundstation1)
	(supports instrument9 spectrograph0)
	(supports instrument9 spectrograph2)
	(calibration_target instrument9 groundstation1)
	(on_board instrument5 satellite1)
	(on_board instrument6 satellite1)
	(on_board instrument7 satellite1)
	(on_board instrument8 satellite1)
	(on_board instrument9 satellite1)
	(power_avail satellite1)
	(pointing satellite1 star0)
)
(:goal (and
	(have_image planet5 thermograph1)
	(have_image phenomenon6 thermograph1)
	(have_image star7 thermograph1)
	(have_image star8 spectrograph0)
))

)
