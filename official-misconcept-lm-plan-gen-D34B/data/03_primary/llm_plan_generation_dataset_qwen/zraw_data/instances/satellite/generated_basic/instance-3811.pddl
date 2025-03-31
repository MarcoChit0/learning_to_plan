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
	image1 - mode
	spectrograph2 - mode
	image0 - mode
	star0 - direction
	groundstation1 - direction
	star3 - direction
	groundstation4 - direction
	groundstation2 - direction
	groundstation5 - direction
	phenomenon6 - direction
	planet7 - direction
	phenomenon8 - direction
	planet9 - direction
)
(:init
	(supports instrument0 image1)
	(calibration_target instrument0 star3)
	(supports instrument1 image0)
	(supports instrument1 spectrograph2)
	(supports instrument1 image1)
	(calibration_target instrument1 groundstation2)
	(supports instrument2 spectrograph2)
	(supports instrument2 image0)
	(supports instrument2 image1)
	(calibration_target instrument2 groundstation4)
	(supports instrument3 spectrograph2)
	(calibration_target instrument3 groundstation4)
	(calibration_target instrument3 groundstation1)
	(supports instrument4 image1)
	(supports instrument4 spectrograph2)
	(calibration_target instrument4 star0)
	(calibration_target instrument4 groundstation5)
	(on_board instrument0 satellite0)
	(on_board instrument1 satellite0)
	(on_board instrument2 satellite0)
	(on_board instrument3 satellite0)
	(on_board instrument4 satellite0)
	(power_avail satellite0)
	(pointing satellite0 phenomenon6)
	(supports instrument5 image1)
	(supports instrument5 image0)
	(calibration_target instrument5 groundstation4)
	(supports instrument6 spectrograph2)
	(supports instrument6 image0)
	(calibration_target instrument6 star3)
	(calibration_target instrument6 groundstation1)
	(on_board instrument5 satellite1)
	(on_board instrument6 satellite1)
	(power_avail satellite1)
	(pointing satellite1 planet9)
	(supports instrument7 spectrograph2)
	(supports instrument7 image1)
	(supports instrument7 image0)
	(calibration_target instrument7 groundstation2)
	(calibration_target instrument7 groundstation4)
	(supports instrument8 image1)
	(supports instrument8 spectrograph2)
	(supports instrument8 image0)
	(calibration_target instrument8 groundstation5)
	(calibration_target instrument8 groundstation2)
	(on_board instrument7 satellite2)
	(on_board instrument8 satellite2)
	(power_avail satellite2)
	(pointing satellite2 star3)
)
(:goal (and
	(pointing satellite1 planet9)
	(have_image phenomenon6 spectrograph2)
	(have_image planet7 image1)
	(have_image phenomenon8 spectrograph2)
	(have_image planet9 image1)
))

)
