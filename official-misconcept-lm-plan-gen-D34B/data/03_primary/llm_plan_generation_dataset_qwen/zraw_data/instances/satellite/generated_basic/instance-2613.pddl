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
	infrared1 - mode
	spectrograph0 - mode
	image2 - mode
	image3 - mode
	star0 - direction
	groundstation1 - direction
	phenomenon2 - direction
)
(:init
	(supports instrument0 image3)
	(supports instrument0 infrared1)
	(supports instrument0 spectrograph0)
	(calibration_target instrument0 groundstation1)
	(supports instrument1 image2)
	(calibration_target instrument1 star0)
	(supports instrument2 infrared1)
	(calibration_target instrument2 star0)
	(supports instrument3 spectrograph0)
	(supports instrument3 image2)
	(calibration_target instrument3 groundstation1)
	(supports instrument4 image2)
	(supports instrument4 image3)
	(calibration_target instrument4 groundstation1)
	(on_board instrument0 satellite0)
	(on_board instrument1 satellite0)
	(on_board instrument2 satellite0)
	(on_board instrument3 satellite0)
	(on_board instrument4 satellite0)
	(power_avail satellite0)
	(pointing satellite0 groundstation1)
	(supports instrument5 spectrograph0)
	(calibration_target instrument5 star0)
	(supports instrument6 image2)
	(calibration_target instrument6 groundstation1)
	(supports instrument7 image2)
	(supports instrument7 infrared1)
	(calibration_target instrument7 groundstation1)
	(supports instrument8 image3)
	(supports instrument8 spectrograph0)
	(supports instrument8 infrared1)
	(calibration_target instrument8 groundstation1)
	(supports instrument9 image3)
	(calibration_target instrument9 groundstation1)
	(on_board instrument5 satellite1)
	(on_board instrument6 satellite1)
	(on_board instrument7 satellite1)
	(on_board instrument8 satellite1)
	(on_board instrument9 satellite1)
	(power_avail satellite1)
	(pointing satellite1 star0)
)
(:goal (and
	(have_image phenomenon2 spectrograph0)
))

)
