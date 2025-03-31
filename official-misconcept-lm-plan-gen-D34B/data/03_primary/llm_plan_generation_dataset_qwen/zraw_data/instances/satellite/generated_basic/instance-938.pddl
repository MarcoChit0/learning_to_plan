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
	instrument8 - instrument
	instrument9 - instrument
	satellite2 - satellite
	instrument10 - instrument
	instrument11 - instrument
	image3 - mode
	spectrograph2 - mode
	image0 - mode
	infrared1 - mode
	groundstation0 - direction
	groundstation2 - direction
	star1 - direction
	star3 - direction
	planet4 - direction
	star5 - direction
	star6 - direction
)
(:init
	(supports instrument0 image0)
	(calibration_target instrument0 groundstation0)
	(supports instrument1 spectrograph2)
	(supports instrument1 image0)
	(supports instrument1 image3)
	(calibration_target instrument1 groundstation0)
	(supports instrument2 infrared1)
	(calibration_target instrument2 star1)
	(supports instrument3 spectrograph2)
	(supports instrument3 infrared1)
	(supports instrument3 image0)
	(calibration_target instrument3 groundstation2)
	(supports instrument4 spectrograph2)
	(supports instrument4 image3)
	(calibration_target instrument4 groundstation2)
	(supports instrument5 spectrograph2)
	(supports instrument5 infrared1)
	(calibration_target instrument5 star3)
	(on_board instrument0 satellite0)
	(on_board instrument1 satellite0)
	(on_board instrument2 satellite0)
	(on_board instrument3 satellite0)
	(on_board instrument4 satellite0)
	(on_board instrument5 satellite0)
	(power_avail satellite0)
	(pointing satellite0 groundstation2)
	(supports instrument6 image0)
	(supports instrument6 infrared1)
	(calibration_target instrument6 groundstation0)
	(supports instrument7 image0)
	(calibration_target instrument7 star1)
	(supports instrument8 spectrograph2)
	(supports instrument8 image0)
	(calibration_target instrument8 groundstation2)
	(supports instrument9 image0)
	(supports instrument9 spectrograph2)
	(supports instrument9 infrared1)
	(calibration_target instrument9 groundstation2)
	(on_board instrument6 satellite1)
	(on_board instrument7 satellite1)
	(on_board instrument8 satellite1)
	(on_board instrument9 satellite1)
	(power_avail satellite1)
	(pointing satellite1 star1)
	(supports instrument10 image3)
	(calibration_target instrument10 star1)
	(supports instrument11 image0)
	(supports instrument11 image3)
	(supports instrument11 infrared1)
	(calibration_target instrument11 star3)
	(on_board instrument10 satellite2)
	(on_board instrument11 satellite2)
	(power_avail satellite2)
	(pointing satellite2 planet4)
)
(:goal (and
	(pointing satellite0 planet4)
	(have_image planet4 infrared1)
	(have_image star5 infrared1)
	(have_image star6 infrared1)
))

)
