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
	thermograph1 - mode
	groundstation4 - direction
	groundstation0 - direction
	star3 - direction
	groundstation1 - direction
	groundstation2 - direction
	groundstation5 - direction
	phenomenon6 - direction
)
(:init
	(supports instrument0 spectrograph0)
	(supports instrument0 thermograph1)
	(calibration_target instrument0 groundstation2)
	(calibration_target instrument0 groundstation4)
	(supports instrument1 spectrograph0)
	(supports instrument1 thermograph1)
	(calibration_target instrument1 groundstation4)
	(calibration_target instrument1 groundstation2)
	(supports instrument2 spectrograph0)
	(supports instrument2 thermograph1)
	(calibration_target instrument2 groundstation2)
	(calibration_target instrument2 groundstation1)
	(supports instrument3 spectrograph0)
	(supports instrument3 thermograph1)
	(calibration_target instrument3 star3)
	(calibration_target instrument3 groundstation2)
	(supports instrument4 thermograph1)
	(supports instrument4 spectrograph0)
	(calibration_target instrument4 groundstation0)
	(on_board instrument0 satellite0)
	(on_board instrument1 satellite0)
	(on_board instrument2 satellite0)
	(on_board instrument3 satellite0)
	(on_board instrument4 satellite0)
	(power_avail satellite0)
	(pointing satellite0 groundstation1)
	(supports instrument5 thermograph1)
	(calibration_target instrument5 star3)
	(supports instrument6 thermograph1)
	(calibration_target instrument6 groundstation1)
	(calibration_target instrument6 star3)
	(supports instrument7 spectrograph0)
	(calibration_target instrument7 groundstation5)
	(supports instrument8 spectrograph0)
	(supports instrument8 thermograph1)
	(calibration_target instrument8 groundstation1)
	(on_board instrument5 satellite1)
	(on_board instrument6 satellite1)
	(on_board instrument7 satellite1)
	(on_board instrument8 satellite1)
	(power_avail satellite1)
	(pointing satellite1 groundstation2)
	(supports instrument9 thermograph1)
	(supports instrument9 spectrograph0)
	(calibration_target instrument9 groundstation5)
	(calibration_target instrument9 groundstation2)
	(on_board instrument9 satellite2)
	(power_avail satellite2)
	(pointing satellite2 groundstation0)
)
(:goal (and
	(pointing satellite2 groundstation2)
	(have_image phenomenon6 spectrograph0)
))

)
