(define (problem strips-sat-x-1)
(:domain satellite)
(:objects
	satellite0 - satellite
	instrument0 - instrument
	instrument1 - instrument
	satellite1 - satellite
	instrument2 - instrument
	instrument3 - instrument
	instrument4 - instrument
	satellite2 - satellite
	instrument5 - instrument
	instrument6 - instrument
	instrument7 - instrument
	instrument8 - instrument
	instrument9 - instrument
	satellite3 - satellite
	instrument10 - instrument
	infrared0 - mode
	thermograph1 - mode
	spectrograph2 - mode
	star1 - direction
	groundstation4 - direction
	groundstation5 - direction
	star3 - direction
	star2 - direction
	groundstation0 - direction
	phenomenon6 - direction
	star7 - direction
	star8 - direction
	star9 - direction
)
(:init
	(supports instrument0 thermograph1)
	(supports instrument0 infrared0)
	(supports instrument0 spectrograph2)
	(calibration_target instrument0 star3)
	(calibration_target instrument0 groundstation0)
	(supports instrument1 infrared0)
	(supports instrument1 spectrograph2)
	(calibration_target instrument1 groundstation5)
	(on_board instrument0 satellite0)
	(on_board instrument1 satellite0)
	(power_avail satellite0)
	(pointing satellite0 groundstation5)
	(supports instrument2 spectrograph2)
	(supports instrument2 infrared0)
	(supports instrument2 thermograph1)
	(calibration_target instrument2 star2)
	(supports instrument3 infrared0)
	(calibration_target instrument3 star3)
	(supports instrument4 spectrograph2)
	(supports instrument4 thermograph1)
	(supports instrument4 infrared0)
	(calibration_target instrument4 groundstation4)
	(calibration_target instrument4 groundstation0)
	(on_board instrument2 satellite1)
	(on_board instrument3 satellite1)
	(on_board instrument4 satellite1)
	(power_avail satellite1)
	(pointing satellite1 groundstation0)
	(supports instrument5 thermograph1)
	(calibration_target instrument5 groundstation4)
	(supports instrument6 thermograph1)
	(supports instrument6 infrared0)
	(supports instrument6 spectrograph2)
	(calibration_target instrument6 star2)
	(supports instrument7 thermograph1)
	(supports instrument7 infrared0)
	(calibration_target instrument7 groundstation5)
	(supports instrument8 spectrograph2)
	(calibration_target instrument8 groundstation0)
	(supports instrument9 spectrograph2)
	(supports instrument9 thermograph1)
	(calibration_target instrument9 star2)
	(calibration_target instrument9 star3)
	(on_board instrument5 satellite2)
	(on_board instrument6 satellite2)
	(on_board instrument7 satellite2)
	(on_board instrument8 satellite2)
	(on_board instrument9 satellite2)
	(power_avail satellite2)
	(pointing satellite2 groundstation0)
	(supports instrument10 spectrograph2)
	(supports instrument10 infrared0)
	(calibration_target instrument10 groundstation0)
	(on_board instrument10 satellite3)
	(power_avail satellite3)
	(pointing satellite3 star1)
)
(:goal (and
	(pointing satellite2 star3)
	(pointing satellite3 star7)
	(have_image phenomenon6 thermograph1)
	(have_image star7 thermograph1)
	(have_image star8 spectrograph2)
	(have_image star9 spectrograph2)
))

)
