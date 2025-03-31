(define (problem strips-sat-x-1)
(:domain satellite)
(:objects
	satellite0 - satellite
	instrument0 - instrument
	instrument1 - instrument
	instrument2 - instrument
	satellite1 - satellite
	instrument3 - instrument
	instrument4 - instrument
	instrument5 - instrument
	satellite2 - satellite
	instrument6 - instrument
	spectrograph1 - mode
	image2 - mode
	image0 - mode
	groundstation1 - direction
	star0 - direction
	phenomenon2 - direction
	star3 - direction
	phenomenon4 - direction
	phenomenon5 - direction
	planet6 - direction
)
(:init
	(supports instrument0 image0)
	(supports instrument0 spectrograph1)
	(supports instrument0 image2)
	(calibration_target instrument0 star0)
	(supports instrument1 image0)
	(calibration_target instrument1 groundstation1)
	(supports instrument2 spectrograph1)
	(calibration_target instrument2 star0)
	(on_board instrument0 satellite0)
	(on_board instrument1 satellite0)
	(on_board instrument2 satellite0)
	(power_avail satellite0)
	(pointing satellite0 phenomenon2)
	(supports instrument3 spectrograph1)
	(calibration_target instrument3 star0)
	(supports instrument4 image0)
	(supports instrument4 image2)
	(calibration_target instrument4 groundstation1)
	(supports instrument5 spectrograph1)
	(supports instrument5 image0)
	(supports instrument5 image2)
	(calibration_target instrument5 star0)
	(on_board instrument3 satellite1)
	(on_board instrument4 satellite1)
	(on_board instrument5 satellite1)
	(power_avail satellite1)
	(pointing satellite1 groundstation1)
	(supports instrument6 spectrograph1)
	(supports instrument6 image0)
	(supports instrument6 image2)
	(calibration_target instrument6 star0)
	(on_board instrument6 satellite2)
	(power_avail satellite2)
	(pointing satellite2 phenomenon4)
)
(:goal (and
	(have_image phenomenon2 spectrograph1)
	(have_image star3 image2)
	(have_image phenomenon4 image2)
	(have_image phenomenon5 image2)
	(have_image planet6 image2)
))

)
