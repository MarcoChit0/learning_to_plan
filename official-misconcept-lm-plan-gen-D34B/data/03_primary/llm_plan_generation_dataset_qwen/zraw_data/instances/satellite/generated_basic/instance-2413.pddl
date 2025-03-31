(define (problem strips-sat-x-1)
(:domain satellite)
(:objects
	satellite0 - satellite
	instrument0 - instrument
	instrument1 - instrument
	instrument2 - instrument
	satellite1 - satellite
	instrument3 - instrument
	satellite2 - satellite
	instrument4 - instrument
	instrument5 - instrument
	satellite3 - satellite
	instrument6 - instrument
	instrument7 - instrument
	instrument8 - instrument
	image1 - mode
	spectrograph0 - mode
	infrared2 - mode
	image3 - mode
	star4 - direction
	groundstation2 - direction
	groundstation3 - direction
	groundstation0 - direction
	groundstation1 - direction
	star5 - direction
	phenomenon6 - direction
)
(:init
	(supports instrument0 image1)
	(supports instrument0 image3)
	(calibration_target instrument0 groundstation2)
	(supports instrument1 infrared2)
	(calibration_target instrument1 groundstation3)
	(supports instrument2 infrared2)
	(supports instrument2 image1)
	(supports instrument2 spectrograph0)
	(calibration_target instrument2 groundstation3)
	(on_board instrument0 satellite0)
	(on_board instrument1 satellite0)
	(on_board instrument2 satellite0)
	(power_avail satellite0)
	(pointing satellite0 groundstation1)
	(supports instrument3 infrared2)
	(supports instrument3 image3)
	(calibration_target instrument3 groundstation3)
	(on_board instrument3 satellite1)
	(power_avail satellite1)
	(pointing satellite1 phenomenon6)
	(supports instrument4 spectrograph0)
	(supports instrument4 infrared2)
	(calibration_target instrument4 groundstation3)
	(supports instrument5 spectrograph0)
	(calibration_target instrument5 groundstation3)
	(on_board instrument4 satellite2)
	(on_board instrument5 satellite2)
	(power_avail satellite2)
	(pointing satellite2 groundstation2)
	(supports instrument6 image1)
	(supports instrument6 spectrograph0)
	(calibration_target instrument6 groundstation0)
	(supports instrument7 image1)
	(supports instrument7 image3)
	(calibration_target instrument7 groundstation1)
	(supports instrument8 image3)
	(supports instrument8 infrared2)
	(supports instrument8 image1)
	(calibration_target instrument8 groundstation1)
	(on_board instrument6 satellite3)
	(on_board instrument7 satellite3)
	(on_board instrument8 satellite3)
	(power_avail satellite3)
	(pointing satellite3 phenomenon6)
)
(:goal (and
	(pointing satellite0 star5)
	(pointing satellite1 groundstation3)
	(pointing satellite2 groundstation1)
	(pointing satellite3 groundstation2)
	(have_image star5 spectrograph0)
	(have_image phenomenon6 image1)
))

)
