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
	instrument8 - instrument
	infrared1 - mode
	spectrograph3 - mode
	image2 - mode
	image0 - mode
	star0 - direction
	star1 - direction
	phenomenon2 - direction
)
(:init
	(supports instrument0 infrared1)
	(calibration_target instrument0 star1)
	(supports instrument1 spectrograph3)
	(supports instrument1 image0)
	(supports instrument1 image2)
	(calibration_target instrument1 star0)
	(supports instrument2 spectrograph3)
	(calibration_target instrument2 star1)
	(supports instrument3 image2)
	(supports instrument3 infrared1)
	(calibration_target instrument3 star1)
	(on_board instrument0 satellite0)
	(on_board instrument1 satellite0)
	(on_board instrument2 satellite0)
	(on_board instrument3 satellite0)
	(power_avail satellite0)
	(pointing satellite0 star0)
	(supports instrument4 image0)
	(supports instrument4 image2)
	(calibration_target instrument4 star1)
	(supports instrument5 spectrograph3)
	(supports instrument5 image0)
	(calibration_target instrument5 star1)
	(supports instrument6 spectrograph3)
	(supports instrument6 image0)
	(calibration_target instrument6 star0)
	(supports instrument7 image0)
	(supports instrument7 spectrograph3)
	(calibration_target instrument7 star1)
	(supports instrument8 image2)
	(supports instrument8 image0)
	(supports instrument8 infrared1)
	(calibration_target instrument8 star1)
	(on_board instrument4 satellite1)
	(on_board instrument5 satellite1)
	(on_board instrument6 satellite1)
	(on_board instrument7 satellite1)
	(on_board instrument8 satellite1)
	(power_avail satellite1)
	(pointing satellite1 star0)
)
(:goal (and
	(pointing satellite0 star1)
	(have_image phenomenon2 infrared1)
))

)
