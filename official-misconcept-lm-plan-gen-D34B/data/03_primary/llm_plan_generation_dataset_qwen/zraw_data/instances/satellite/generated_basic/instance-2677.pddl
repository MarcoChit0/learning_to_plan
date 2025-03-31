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
	image0 - mode
	infrared2 - mode
	spectrograph3 - mode
	thermograph1 - mode
	groundstation1 - direction
	star3 - direction
	star2 - direction
	star0 - direction
	phenomenon4 - direction
)
(:init
	(supports instrument0 image0)
	(supports instrument0 spectrograph3)
	(calibration_target instrument0 star0)
	(supports instrument1 spectrograph3)
	(supports instrument1 thermograph1)
	(calibration_target instrument1 star3)
	(supports instrument2 infrared2)
	(supports instrument2 spectrograph3)
	(supports instrument2 thermograph1)
	(calibration_target instrument2 star2)
	(supports instrument3 image0)
	(supports instrument3 infrared2)
	(calibration_target instrument3 star3)
	(supports instrument4 image0)
	(calibration_target instrument4 star0)
	(on_board instrument0 satellite0)
	(on_board instrument1 satellite0)
	(on_board instrument2 satellite0)
	(on_board instrument3 satellite0)
	(on_board instrument4 satellite0)
	(power_avail satellite0)
	(pointing satellite0 star3)
	(supports instrument5 image0)
	(calibration_target instrument5 star3)
	(supports instrument6 thermograph1)
	(calibration_target instrument6 star2)
	(supports instrument7 image0)
	(supports instrument7 spectrograph3)
	(calibration_target instrument7 star2)
	(supports instrument8 thermograph1)
	(calibration_target instrument8 star0)
	(on_board instrument5 satellite1)
	(on_board instrument6 satellite1)
	(on_board instrument7 satellite1)
	(on_board instrument8 satellite1)
	(power_avail satellite1)
	(pointing satellite1 star0)
)
(:goal (and
	(pointing satellite0 star2)
	(have_image phenomenon4 spectrograph3)
))

)
