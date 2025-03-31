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
	instrument9 - instrument
	instrument10 - instrument
	satellite2 - satellite
	instrument11 - instrument
	spectrograph2 - mode
	image1 - mode
	image0 - mode
	groundstation0 - direction
	groundstation1 - direction
	phenomenon2 - direction
)
(:init
	(supports instrument0 spectrograph2)
	(supports instrument0 image1)
	(calibration_target instrument0 groundstation0)
	(supports instrument1 image1)
	(supports instrument1 spectrograph2)
	(supports instrument1 image0)
	(calibration_target instrument1 groundstation0)
	(supports instrument2 image0)
	(supports instrument2 spectrograph2)
	(calibration_target instrument2 groundstation0)
	(supports instrument3 spectrograph2)
	(calibration_target instrument3 groundstation0)
	(supports instrument4 image1)
	(supports instrument4 image0)
	(calibration_target instrument4 groundstation0)
	(on_board instrument0 satellite0)
	(on_board instrument1 satellite0)
	(on_board instrument2 satellite0)
	(on_board instrument3 satellite0)
	(on_board instrument4 satellite0)
	(power_avail satellite0)
	(pointing satellite0 groundstation0)
	(supports instrument5 image0)
	(supports instrument5 spectrograph2)
	(supports instrument5 image1)
	(calibration_target instrument5 groundstation1)
	(supports instrument6 spectrograph2)
	(supports instrument6 image0)
	(calibration_target instrument6 groundstation0)
	(supports instrument7 image0)
	(calibration_target instrument7 groundstation0)
	(supports instrument8 image1)
	(calibration_target instrument8 groundstation1)
	(supports instrument9 spectrograph2)
	(supports instrument9 image1)
	(calibration_target instrument9 groundstation0)
	(supports instrument10 spectrograph2)
	(supports instrument10 image1)
	(calibration_target instrument10 groundstation1)
	(on_board instrument5 satellite1)
	(on_board instrument6 satellite1)
	(on_board instrument7 satellite1)
	(on_board instrument8 satellite1)
	(on_board instrument9 satellite1)
	(on_board instrument10 satellite1)
	(power_avail satellite1)
	(pointing satellite1 phenomenon2)
	(supports instrument11 image0)
	(calibration_target instrument11 groundstation1)
	(on_board instrument11 satellite2)
	(power_avail satellite2)
	(pointing satellite2 groundstation1)
)
(:goal (and
	(have_image phenomenon2 image1)
))

)
