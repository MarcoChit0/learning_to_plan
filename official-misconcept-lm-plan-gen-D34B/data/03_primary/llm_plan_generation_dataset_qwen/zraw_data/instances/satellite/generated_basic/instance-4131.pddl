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
	satellite2 - satellite
	instrument8 - instrument
	instrument9 - instrument
	spectrograph1 - mode
	thermograph0 - mode
	groundstation2 - direction
	star0 - direction
	star5 - direction
	groundstation4 - direction
	groundstation1 - direction
	groundstation3 - direction
	phenomenon6 - direction
)
(:init
	(supports instrument0 spectrograph1)
	(calibration_target instrument0 star0)
	(calibration_target instrument0 groundstation2)
	(supports instrument1 thermograph0)
	(supports instrument1 spectrograph1)
	(calibration_target instrument1 groundstation1)
	(supports instrument2 thermograph0)
	(supports instrument2 spectrograph1)
	(calibration_target instrument2 groundstation1)
	(calibration_target instrument2 star0)
	(supports instrument3 thermograph0)
	(calibration_target instrument3 star0)
	(calibration_target instrument3 star5)
	(supports instrument4 thermograph0)
	(supports instrument4 spectrograph1)
	(calibration_target instrument4 groundstation1)
	(calibration_target instrument4 star5)
	(supports instrument5 thermograph0)
	(supports instrument5 spectrograph1)
	(calibration_target instrument5 groundstation4)
	(calibration_target instrument5 groundstation3)
	(on_board instrument0 satellite0)
	(on_board instrument1 satellite0)
	(on_board instrument2 satellite0)
	(on_board instrument3 satellite0)
	(on_board instrument4 satellite0)
	(on_board instrument5 satellite0)
	(power_avail satellite0)
	(pointing satellite0 star5)
	(supports instrument6 thermograph0)
	(calibration_target instrument6 groundstation4)
	(calibration_target instrument6 star5)
	(supports instrument7 thermograph0)
	(supports instrument7 spectrograph1)
	(calibration_target instrument7 groundstation1)
	(calibration_target instrument7 groundstation3)
	(on_board instrument6 satellite1)
	(on_board instrument7 satellite1)
	(power_avail satellite1)
	(pointing satellite1 groundstation3)
	(supports instrument8 thermograph0)
	(calibration_target instrument8 groundstation1)
	(calibration_target instrument8 groundstation4)
	(supports instrument9 thermograph0)
	(supports instrument9 spectrograph1)
	(calibration_target instrument9 groundstation3)
	(on_board instrument8 satellite2)
	(on_board instrument9 satellite2)
	(power_avail satellite2)
	(pointing satellite2 groundstation1)
)
(:goal (and
	(pointing satellite0 groundstation2)
	(have_image phenomenon6 thermograph0)
))

)
