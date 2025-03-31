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
	instrument7 - instrument
	satellite3 - satellite
	instrument8 - instrument
	image1 - mode
	image2 - mode
	image3 - mode
	spectrograph0 - mode
	star3 - direction
	star2 - direction
	groundstation4 - direction
	groundstation1 - direction
	star0 - direction
	phenomenon5 - direction
	phenomenon6 - direction
)
(:init
	(supports instrument0 image3)
	(supports instrument0 spectrograph0)
	(calibration_target instrument0 star3)
	(supports instrument1 image1)
	(calibration_target instrument1 star3)
	(on_board instrument0 satellite0)
	(on_board instrument1 satellite0)
	(power_avail satellite0)
	(pointing satellite0 groundstation4)
	(supports instrument2 image1)
	(calibration_target instrument2 star3)
	(supports instrument3 image2)
	(supports instrument3 spectrograph0)
	(supports instrument3 image3)
	(calibration_target instrument3 star2)
	(on_board instrument2 satellite1)
	(on_board instrument3 satellite1)
	(power_avail satellite1)
	(pointing satellite1 groundstation4)
	(supports instrument4 spectrograph0)
	(calibration_target instrument4 groundstation1)
	(supports instrument5 image1)
	(calibration_target instrument5 groundstation4)
	(supports instrument6 spectrograph0)
	(supports instrument6 image1)
	(calibration_target instrument6 star0)
	(supports instrument7 image3)
	(supports instrument7 spectrograph0)
	(calibration_target instrument7 groundstation1)
	(on_board instrument4 satellite2)
	(on_board instrument5 satellite2)
	(on_board instrument6 satellite2)
	(on_board instrument7 satellite2)
	(power_avail satellite2)
	(pointing satellite2 star2)
	(supports instrument8 image3)
	(supports instrument8 image1)
	(supports instrument8 image2)
	(calibration_target instrument8 star0)
	(on_board instrument8 satellite3)
	(power_avail satellite3)
	(pointing satellite3 groundstation1)
)
(:goal (and
	(pointing satellite0 groundstation4)
	(pointing satellite2 star2)
	(have_image phenomenon5 spectrograph0)
	(have_image phenomenon6 image1)
))

)
