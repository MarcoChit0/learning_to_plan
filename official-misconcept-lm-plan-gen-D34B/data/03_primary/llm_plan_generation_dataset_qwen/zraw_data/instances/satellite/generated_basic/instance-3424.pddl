(define (problem strips-sat-x-1)
(:domain satellite)
(:objects
	satellite0 - satellite
	instrument0 - instrument
	satellite1 - satellite
	instrument1 - instrument
	satellite2 - satellite
	instrument2 - instrument
	instrument3 - instrument
	instrument4 - instrument
	satellite3 - satellite
	instrument5 - instrument
	instrument6 - instrument
	instrument7 - instrument
	instrument8 - instrument
	instrument9 - instrument
	thermograph2 - mode
	image0 - mode
	spectrograph1 - mode
	groundstation0 - direction
	groundstation1 - direction
	star3 - direction
	groundstation2 - direction
	phenomenon4 - direction
)
(:init
	(supports instrument0 image0)
	(calibration_target instrument0 groundstation1)
	(on_board instrument0 satellite0)
	(power_avail satellite0)
	(pointing satellite0 star3)
	(supports instrument1 spectrograph1)
	(supports instrument1 image0)
	(supports instrument1 thermograph2)
	(calibration_target instrument1 groundstation2)
	(on_board instrument1 satellite1)
	(power_avail satellite1)
	(pointing satellite1 groundstation0)
	(supports instrument2 image0)
	(supports instrument2 spectrograph1)
	(calibration_target instrument2 groundstation1)
	(supports instrument3 spectrograph1)
	(supports instrument3 image0)
	(supports instrument3 thermograph2)
	(calibration_target instrument3 groundstation2)
	(supports instrument4 spectrograph1)
	(supports instrument4 thermograph2)
	(calibration_target instrument4 groundstation1)
	(on_board instrument2 satellite2)
	(on_board instrument3 satellite2)
	(on_board instrument4 satellite2)
	(power_avail satellite2)
	(pointing satellite2 groundstation0)
	(supports instrument5 spectrograph1)
	(calibration_target instrument5 star3)
	(supports instrument6 spectrograph1)
	(supports instrument6 image0)
	(supports instrument6 thermograph2)
	(calibration_target instrument6 groundstation2)
	(supports instrument7 spectrograph1)
	(supports instrument7 thermograph2)
	(supports instrument7 image0)
	(calibration_target instrument7 groundstation2)
	(supports instrument8 image0)
	(calibration_target instrument8 groundstation2)
	(supports instrument9 spectrograph1)
	(supports instrument9 image0)
	(supports instrument9 thermograph2)
	(calibration_target instrument9 groundstation2)
	(on_board instrument5 satellite3)
	(on_board instrument6 satellite3)
	(on_board instrument7 satellite3)
	(on_board instrument8 satellite3)
	(on_board instrument9 satellite3)
	(power_avail satellite3)
	(pointing satellite3 groundstation2)
)
(:goal (and
	(pointing satellite1 star3)
	(pointing satellite3 groundstation2)
	(have_image phenomenon4 thermograph2)
))

)
