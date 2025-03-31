(define (problem strips-sat-x-1)
(:domain satellite)
(:objects
	satellite0 - satellite
	instrument0 - instrument
	instrument1 - instrument
	satellite1 - satellite
	instrument2 - instrument
	satellite2 - satellite
	instrument3 - instrument
	instrument4 - instrument
	instrument5 - instrument
	satellite3 - satellite
	instrument6 - instrument
	instrument7 - instrument
	instrument8 - instrument
	instrument9 - instrument
	instrument10 - instrument
	thermograph1 - mode
	spectrograph0 - mode
	thermograph2 - mode
	groundstation0 - direction
	groundstation1 - direction
	groundstation2 - direction
	planet3 - direction
	star4 - direction
)
(:init
	(supports instrument0 thermograph2)
	(supports instrument0 spectrograph0)
	(calibration_target instrument0 groundstation1)
	(supports instrument1 spectrograph0)
	(supports instrument1 thermograph2)
	(calibration_target instrument1 groundstation2)
	(on_board instrument0 satellite0)
	(on_board instrument1 satellite0)
	(power_avail satellite0)
	(pointing satellite0 star4)
	(supports instrument2 thermograph2)
	(calibration_target instrument2 groundstation2)
	(on_board instrument2 satellite1)
	(power_avail satellite1)
	(pointing satellite1 planet3)
	(supports instrument3 thermograph1)
	(calibration_target instrument3 groundstation2)
	(supports instrument4 thermograph2)
	(supports instrument4 spectrograph0)
	(supports instrument4 thermograph1)
	(calibration_target instrument4 groundstation2)
	(supports instrument5 thermograph2)
	(supports instrument5 thermograph1)
	(supports instrument5 spectrograph0)
	(calibration_target instrument5 groundstation0)
	(on_board instrument3 satellite2)
	(on_board instrument4 satellite2)
	(on_board instrument5 satellite2)
	(power_avail satellite2)
	(pointing satellite2 groundstation1)
	(supports instrument6 spectrograph0)
	(calibration_target instrument6 groundstation0)
	(supports instrument7 spectrograph0)
	(supports instrument7 thermograph2)
	(calibration_target instrument7 groundstation0)
	(supports instrument8 spectrograph0)
	(calibration_target instrument8 groundstation1)
	(supports instrument9 spectrograph0)
	(supports instrument9 thermograph1)
	(supports instrument9 thermograph2)
	(calibration_target instrument9 groundstation2)
	(supports instrument10 thermograph1)
	(calibration_target instrument10 groundstation2)
	(on_board instrument6 satellite3)
	(on_board instrument7 satellite3)
	(on_board instrument8 satellite3)
	(on_board instrument9 satellite3)
	(on_board instrument10 satellite3)
	(power_avail satellite3)
	(pointing satellite3 groundstation2)
)
(:goal (and
	(pointing satellite0 groundstation2)
	(pointing satellite1 star4)
	(pointing satellite3 planet3)
	(have_image planet3 thermograph2)
	(have_image star4 spectrograph0)
))

)
