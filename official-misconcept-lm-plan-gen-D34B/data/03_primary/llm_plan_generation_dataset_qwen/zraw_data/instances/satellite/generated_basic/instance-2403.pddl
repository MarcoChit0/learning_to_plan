(define (problem strips-sat-x-1)
(:domain satellite)
(:objects
	satellite0 - satellite
	instrument0 - instrument
	instrument1 - instrument
	instrument2 - instrument
	instrument3 - instrument
	satellite1 - satellite
	instrument4 - instrument
	instrument5 - instrument
	instrument6 - instrument
	instrument7 - instrument
	satellite2 - satellite
	instrument8 - instrument
	instrument9 - instrument
	satellite3 - satellite
	instrument10 - instrument
	spectrograph1 - mode
	spectrograph0 - mode
	image3 - mode
	image2 - mode
	groundstation2 - direction
	groundstation1 - direction
	groundstation4 - direction
	star0 - direction
	star3 - direction
	phenomenon5 - direction
	planet6 - direction
)
(:init
	(supports instrument0 spectrograph0)
	(supports instrument0 spectrograph1)
	(calibration_target instrument0 groundstation1)
	(supports instrument1 image2)
	(calibration_target instrument1 star0)
	(supports instrument2 image3)
	(calibration_target instrument2 groundstation2)
	(supports instrument3 image3)
	(supports instrument3 spectrograph1)
	(supports instrument3 image2)
	(calibration_target instrument3 star3)
	(on_board instrument0 satellite0)
	(on_board instrument1 satellite0)
	(on_board instrument2 satellite0)
	(on_board instrument3 satellite0)
	(power_avail satellite0)
	(pointing satellite0 groundstation4)
	(supports instrument4 spectrograph1)
	(calibration_target instrument4 groundstation1)
	(supports instrument5 spectrograph1)
	(supports instrument5 spectrograph0)
	(supports instrument5 image2)
	(calibration_target instrument5 star0)
	(supports instrument6 image2)
	(supports instrument6 image3)
	(supports instrument6 spectrograph1)
	(calibration_target instrument6 star0)
	(supports instrument7 spectrograph1)
	(supports instrument7 image3)
	(supports instrument7 image2)
	(calibration_target instrument7 groundstation4)
	(on_board instrument4 satellite1)
	(on_board instrument5 satellite1)
	(on_board instrument6 satellite1)
	(on_board instrument7 satellite1)
	(power_avail satellite1)
	(pointing satellite1 star0)
	(supports instrument8 spectrograph1)
	(supports instrument8 image2)
	(calibration_target instrument8 star0)
	(supports instrument9 image3)
	(supports instrument9 spectrograph1)
	(calibration_target instrument9 star3)
	(on_board instrument8 satellite2)
	(on_board instrument9 satellite2)
	(power_avail satellite2)
	(pointing satellite2 phenomenon5)
	(supports instrument10 spectrograph1)
	(supports instrument10 image2)
	(calibration_target instrument10 star3)
	(on_board instrument10 satellite3)
	(power_avail satellite3)
	(pointing satellite3 phenomenon5)
)
(:goal (and
	(pointing satellite0 groundstation2)
	(pointing satellite1 star3)
	(have_image phenomenon5 spectrograph0)
	(have_image planet6 spectrograph0)
))

)
