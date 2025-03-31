(define (problem strips-sat-x-1)
(:domain satellite)
(:objects
	satellite0 - satellite
	instrument0 - instrument
	instrument1 - instrument
	satellite1 - satellite
	instrument2 - instrument
	instrument3 - instrument
	instrument4 - instrument
	instrument5 - instrument
	instrument6 - instrument
	satellite2 - satellite
	instrument7 - instrument
	satellite3 - satellite
	instrument8 - instrument
	image0 - mode
	spectrograph3 - mode
	spectrograph2 - mode
	infrared1 - mode
	groundstation4 - direction
	groundstation3 - direction
	star2 - direction
	star1 - direction
	star0 - direction
	phenomenon5 - direction
	phenomenon6 - direction
	phenomenon7 - direction
)
(:init
	(supports instrument0 spectrograph3)
	(calibration_target instrument0 star0)
	(supports instrument1 infrared1)
	(calibration_target instrument1 star1)
	(on_board instrument0 satellite0)
	(on_board instrument1 satellite0)
	(power_avail satellite0)
	(pointing satellite0 phenomenon6)
	(supports instrument2 image0)
	(calibration_target instrument2 star0)
	(supports instrument3 infrared1)
	(supports instrument3 image0)
	(calibration_target instrument3 groundstation3)
	(supports instrument4 image0)
	(calibration_target instrument4 star2)
	(supports instrument5 spectrograph2)
	(calibration_target instrument5 star0)
	(supports instrument6 image0)
	(supports instrument6 spectrograph3)
	(calibration_target instrument6 star1)
	(on_board instrument2 satellite1)
	(on_board instrument3 satellite1)
	(on_board instrument4 satellite1)
	(on_board instrument5 satellite1)
	(on_board instrument6 satellite1)
	(power_avail satellite1)
	(pointing satellite1 star1)
	(supports instrument7 infrared1)
	(calibration_target instrument7 star1)
	(on_board instrument7 satellite2)
	(power_avail satellite2)
	(pointing satellite2 phenomenon5)
	(supports instrument8 spectrograph2)
	(supports instrument8 spectrograph3)
	(supports instrument8 image0)
	(calibration_target instrument8 star0)
	(on_board instrument8 satellite3)
	(power_avail satellite3)
	(pointing satellite3 phenomenon5)
)
(:goal (and
	(pointing satellite0 phenomenon6)
	(have_image phenomenon5 spectrograph3)
	(have_image phenomenon6 spectrograph2)
	(have_image phenomenon7 spectrograph2)
))

)
