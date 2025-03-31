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
	image3 - mode
	image0 - mode
	spectrograph1 - mode
	image2 - mode
	groundstation3 - direction
	star1 - direction
	groundstation2 - direction
	star0 - direction
	phenomenon4 - direction
	planet5 - direction
	star6 - direction
)
(:init
	(supports instrument0 image0)
	(supports instrument0 spectrograph1)
	(calibration_target instrument0 groundstation3)
	(supports instrument1 image2)
	(calibration_target instrument1 groundstation2)
	(on_board instrument0 satellite0)
	(on_board instrument1 satellite0)
	(power_avail satellite0)
	(pointing satellite0 phenomenon4)
	(supports instrument2 spectrograph1)
	(calibration_target instrument2 star0)
	(supports instrument3 spectrograph1)
	(supports instrument3 image0)
	(supports instrument3 image3)
	(calibration_target instrument3 star0)
	(on_board instrument2 satellite1)
	(on_board instrument3 satellite1)
	(power_avail satellite1)
	(pointing satellite1 planet5)
	(supports instrument4 image2)
	(supports instrument4 image3)
	(supports instrument4 spectrograph1)
	(calibration_target instrument4 star1)
	(supports instrument5 image2)
	(calibration_target instrument5 groundstation2)
	(supports instrument6 image3)
	(supports instrument6 spectrograph1)
	(supports instrument6 image0)
	(calibration_target instrument6 star0)
	(on_board instrument4 satellite2)
	(on_board instrument5 satellite2)
	(on_board instrument6 satellite2)
	(power_avail satellite2)
	(pointing satellite2 star1)
)
(:goal (and
	(pointing satellite0 star1)
	(pointing satellite1 groundstation2)
	(have_image phenomenon4 image3)
	(have_image planet5 image0)
	(have_image star6 spectrograph1)
))

)
