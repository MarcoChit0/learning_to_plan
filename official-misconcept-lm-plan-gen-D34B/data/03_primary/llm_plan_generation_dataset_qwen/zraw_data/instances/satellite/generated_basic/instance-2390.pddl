(define (problem strips-sat-x-1)
(:domain satellite)
(:objects
	satellite0 - satellite
	instrument0 - instrument
	instrument1 - instrument
	satellite1 - satellite
	instrument2 - instrument
	instrument3 - instrument
	satellite2 - satellite
	instrument4 - instrument
	instrument5 - instrument
	instrument6 - instrument
	satellite3 - satellite
	instrument7 - instrument
	instrument8 - instrument
	instrument9 - instrument
	instrument10 - instrument
	thermograph3 - mode
	infrared0 - mode
	spectrograph2 - mode
	spectrograph1 - mode
	star2 - direction
	groundstation4 - direction
	groundstation0 - direction
	groundstation3 - direction
	star1 - direction
	star5 - direction
	phenomenon6 - direction
)
(:init
	(supports instrument0 spectrograph1)
	(calibration_target instrument0 groundstation0)
	(supports instrument1 thermograph3)
	(supports instrument1 spectrograph1)
	(calibration_target instrument1 groundstation4)
	(on_board instrument0 satellite0)
	(on_board instrument1 satellite0)
	(power_avail satellite0)
	(pointing satellite0 phenomenon6)
	(supports instrument2 infrared0)
	(supports instrument2 spectrograph1)
	(supports instrument2 spectrograph2)
	(calibration_target instrument2 star1)
	(supports instrument3 thermograph3)
	(supports instrument3 spectrograph2)
	(supports instrument3 infrared0)
	(calibration_target instrument3 groundstation0)
	(on_board instrument2 satellite1)
	(on_board instrument3 satellite1)
	(power_avail satellite1)
	(pointing satellite1 star5)
	(supports instrument4 infrared0)
	(supports instrument4 spectrograph1)
	(calibration_target instrument4 groundstation0)
	(supports instrument5 spectrograph2)
	(calibration_target instrument5 star1)
	(supports instrument6 thermograph3)
	(calibration_target instrument6 groundstation3)
	(on_board instrument4 satellite2)
	(on_board instrument5 satellite2)
	(on_board instrument6 satellite2)
	(power_avail satellite2)
	(pointing satellite2 groundstation0)
	(supports instrument7 spectrograph1)
	(supports instrument7 spectrograph2)
	(supports instrument7 thermograph3)
	(calibration_target instrument7 groundstation3)
	(supports instrument8 thermograph3)
	(supports instrument8 spectrograph1)
	(calibration_target instrument8 star1)
	(supports instrument9 thermograph3)
	(supports instrument9 spectrograph1)
	(supports instrument9 spectrograph2)
	(calibration_target instrument9 groundstation3)
	(supports instrument10 thermograph3)
	(supports instrument10 spectrograph2)
	(supports instrument10 spectrograph1)
	(calibration_target instrument10 star1)
	(on_board instrument7 satellite3)
	(on_board instrument8 satellite3)
	(on_board instrument9 satellite3)
	(on_board instrument10 satellite3)
	(power_avail satellite3)
	(pointing satellite3 star2)
)
(:goal (and
	(pointing satellite1 groundstation4)
	(pointing satellite2 groundstation0)
	(have_image star5 spectrograph1)
	(have_image phenomenon6 spectrograph1)
))

)
