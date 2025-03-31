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
	satellite2 - satellite
	instrument9 - instrument
	spectrograph0 - mode
	thermograph2 - mode
	spectrograph1 - mode
	star1 - direction
	star3 - direction
	groundstation0 - direction
	groundstation2 - direction
	planet4 - direction
	planet5 - direction
	phenomenon6 - direction
	phenomenon7 - direction
	planet8 - direction
)
(:init
	(supports instrument0 spectrograph0)
	(calibration_target instrument0 groundstation0)
	(supports instrument1 thermograph2)
	(supports instrument1 spectrograph1)
	(supports instrument1 spectrograph0)
	(calibration_target instrument1 star1)
	(supports instrument2 thermograph2)
	(supports instrument2 spectrograph1)
	(supports instrument2 spectrograph0)
	(calibration_target instrument2 star1)
	(supports instrument3 thermograph2)
	(supports instrument3 spectrograph1)
	(supports instrument3 spectrograph0)
	(calibration_target instrument3 star1)
	(supports instrument4 spectrograph0)
	(supports instrument4 thermograph2)
	(calibration_target instrument4 groundstation2)
	(on_board instrument0 satellite0)
	(on_board instrument1 satellite0)
	(on_board instrument2 satellite0)
	(on_board instrument3 satellite0)
	(on_board instrument4 satellite0)
	(power_avail satellite0)
	(pointing satellite0 groundstation0)
	(supports instrument5 spectrograph1)
	(supports instrument5 spectrograph0)
	(calibration_target instrument5 star3)
	(supports instrument6 spectrograph0)
	(calibration_target instrument6 groundstation0)
	(supports instrument7 spectrograph0)
	(calibration_target instrument7 groundstation0)
	(supports instrument8 spectrograph0)
	(supports instrument8 thermograph2)
	(supports instrument8 spectrograph1)
	(calibration_target instrument8 groundstation2)
	(on_board instrument5 satellite1)
	(on_board instrument6 satellite1)
	(on_board instrument7 satellite1)
	(on_board instrument8 satellite1)
	(power_avail satellite1)
	(pointing satellite1 phenomenon7)
	(supports instrument9 spectrograph0)
	(calibration_target instrument9 groundstation2)
	(on_board instrument9 satellite2)
	(power_avail satellite2)
	(pointing satellite2 star1)
)
(:goal (and
	(have_image planet4 thermograph2)
	(have_image planet5 spectrograph0)
	(have_image phenomenon6 spectrograph0)
	(have_image phenomenon7 spectrograph0)
	(have_image planet8 spectrograph0)
))

)
