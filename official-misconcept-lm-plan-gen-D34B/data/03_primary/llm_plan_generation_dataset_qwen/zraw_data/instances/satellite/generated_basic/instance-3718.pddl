(define (problem strips-sat-x-1)
(:domain satellite)
(:objects
	satellite0 - satellite
	instrument0 - instrument
	satellite1 - satellite
	instrument1 - instrument
	instrument2 - instrument
	instrument3 - instrument
	instrument4 - instrument
	satellite2 - satellite
	instrument5 - instrument
	instrument6 - instrument
	instrument7 - instrument
	instrument8 - instrument
	instrument9 - instrument
	thermograph1 - mode
	image0 - mode
	spectrograph2 - mode
	star1 - direction
	groundstation5 - direction
	groundstation0 - direction
	star4 - direction
	groundstation2 - direction
	star3 - direction
	phenomenon6 - direction
)
(:init
	(supports instrument0 spectrograph2)
	(supports instrument0 thermograph1)
	(supports instrument0 image0)
	(calibration_target instrument0 star3)
	(calibration_target instrument0 groundstation5)
	(on_board instrument0 satellite0)
	(power_avail satellite0)
	(pointing satellite0 star1)
	(supports instrument1 thermograph1)
	(supports instrument1 image0)
	(supports instrument1 spectrograph2)
	(calibration_target instrument1 groundstation2)
	(supports instrument2 image0)
	(supports instrument2 thermograph1)
	(calibration_target instrument2 groundstation5)
	(supports instrument3 thermograph1)
	(supports instrument3 image0)
	(calibration_target instrument3 star1)
	(supports instrument4 thermograph1)
	(calibration_target instrument4 star4)
	(calibration_target instrument4 star3)
	(on_board instrument1 satellite1)
	(on_board instrument2 satellite1)
	(on_board instrument3 satellite1)
	(on_board instrument4 satellite1)
	(power_avail satellite1)
	(pointing satellite1 groundstation2)
	(supports instrument5 thermograph1)
	(supports instrument5 spectrograph2)
	(supports instrument5 image0)
	(calibration_target instrument5 star3)
	(supports instrument6 image0)
	(supports instrument6 spectrograph2)
	(supports instrument6 thermograph1)
	(calibration_target instrument6 star4)
	(calibration_target instrument6 groundstation2)
	(supports instrument7 image0)
	(supports instrument7 spectrograph2)
	(supports instrument7 thermograph1)
	(calibration_target instrument7 groundstation5)
	(supports instrument8 spectrograph2)
	(supports instrument8 thermograph1)
	(supports instrument8 image0)
	(calibration_target instrument8 star4)
	(calibration_target instrument8 groundstation0)
	(supports instrument9 thermograph1)
	(calibration_target instrument9 star3)
	(calibration_target instrument9 groundstation2)
	(on_board instrument5 satellite2)
	(on_board instrument6 satellite2)
	(on_board instrument7 satellite2)
	(on_board instrument8 satellite2)
	(on_board instrument9 satellite2)
	(power_avail satellite2)
	(pointing satellite2 groundstation0)
)
(:goal (and
	(pointing satellite0 phenomenon6)
	(have_image phenomenon6 spectrograph2)
))

)
