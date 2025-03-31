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
	satellite2 - satellite
	instrument7 - instrument
	instrument8 - instrument
	instrument9 - instrument
	instrument10 - instrument
	infrared1 - mode
	image0 - mode
	spectrograph2 - mode
	star5 - direction
	groundstation4 - direction
	star1 - direction
	groundstation0 - direction
	star3 - direction
	groundstation2 - direction
	star6 - direction
)
(:init
	(supports instrument0 infrared1)
	(supports instrument0 spectrograph2)
	(supports instrument0 image0)
	(calibration_target instrument0 groundstation0)
	(supports instrument1 spectrograph2)
	(calibration_target instrument1 star3)
	(calibration_target instrument1 groundstation4)
	(supports instrument2 image0)
	(supports instrument2 spectrograph2)
	(supports instrument2 infrared1)
	(calibration_target instrument2 groundstation0)
	(supports instrument3 spectrograph2)
	(calibration_target instrument3 star3)
	(supports instrument4 image0)
	(supports instrument4 infrared1)
	(supports instrument4 spectrograph2)
	(calibration_target instrument4 groundstation0)
	(on_board instrument0 satellite0)
	(on_board instrument1 satellite0)
	(on_board instrument2 satellite0)
	(on_board instrument3 satellite0)
	(on_board instrument4 satellite0)
	(power_avail satellite0)
	(pointing satellite0 star3)
	(supports instrument5 spectrograph2)
	(calibration_target instrument5 star3)
	(calibration_target instrument5 groundstation2)
	(supports instrument6 spectrograph2)
	(supports instrument6 infrared1)
	(calibration_target instrument6 star1)
	(calibration_target instrument6 groundstation4)
	(on_board instrument5 satellite1)
	(on_board instrument6 satellite1)
	(power_avail satellite1)
	(pointing satellite1 groundstation4)
	(supports instrument7 infrared1)
	(supports instrument7 image0)
	(calibration_target instrument7 groundstation4)
	(supports instrument8 image0)
	(calibration_target instrument8 groundstation0)
	(calibration_target instrument8 star1)
	(supports instrument9 image0)
	(calibration_target instrument9 star3)
	(calibration_target instrument9 groundstation0)
	(supports instrument10 spectrograph2)
	(supports instrument10 image0)
	(supports instrument10 infrared1)
	(calibration_target instrument10 groundstation2)
	(on_board instrument7 satellite2)
	(on_board instrument8 satellite2)
	(on_board instrument9 satellite2)
	(on_board instrument10 satellite2)
	(power_avail satellite2)
	(pointing satellite2 star3)
)
(:goal (and
	(pointing satellite1 star3)
	(have_image star6 spectrograph2)
))

)
