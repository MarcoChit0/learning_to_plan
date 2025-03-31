(define (problem strips-sat-x-1)
(:domain satellite)
(:objects
	satellite0 - satellite
	instrument0 - instrument
	satellite1 - satellite
	instrument1 - instrument
	instrument2 - instrument
	instrument3 - instrument
	instrument4 - instrument
	satellite2 - satellite
	instrument5 - instrument
	satellite3 - satellite
	instrument6 - instrument
	instrument7 - instrument
	instrument8 - instrument
	spectrograph3 - mode
	thermograph0 - mode
	spectrograph1 - mode
	image2 - mode
	groundstation1 - direction
	groundstation2 - direction
	star0 - direction
	groundstation3 - direction
	star4 - direction
	planet5 - direction
	star6 - direction
	phenomenon7 - direction
	phenomenon8 - direction
	planet9 - direction
)
(:init
	(supports instrument0 spectrograph3)
	(calibration_target instrument0 groundstation1)
	(on_board instrument0 satellite0)
	(power_avail satellite0)
	(pointing satellite0 phenomenon7)
	(supports instrument1 spectrograph1)
	(calibration_target instrument1 groundstation3)
	(supports instrument2 spectrograph1)
	(calibration_target instrument2 star0)
	(supports instrument3 spectrograph3)
	(calibration_target instrument3 groundstation3)
	(supports instrument4 spectrograph1)
	(supports instrument4 thermograph0)
	(calibration_target instrument4 groundstation3)
	(on_board instrument1 satellite1)
	(on_board instrument2 satellite1)
	(on_board instrument3 satellite1)
	(on_board instrument4 satellite1)
	(power_avail satellite1)
	(pointing satellite1 phenomenon8)
	(supports instrument5 spectrograph1)
	(supports instrument5 thermograph0)
	(calibration_target instrument5 groundstation1)
	(on_board instrument5 satellite2)
	(power_avail satellite2)
	(pointing satellite2 star4)
	(supports instrument6 spectrograph1)
	(supports instrument6 spectrograph3)
	(supports instrument6 thermograph0)
	(calibration_target instrument6 groundstation2)
	(supports instrument7 spectrograph3)
	(calibration_target instrument7 star0)
	(supports instrument8 spectrograph1)
	(supports instrument8 image2)
	(calibration_target instrument8 groundstation3)
	(on_board instrument6 satellite3)
	(on_board instrument7 satellite3)
	(on_board instrument8 satellite3)
	(power_avail satellite3)
	(pointing satellite3 planet9)
)
(:goal (and
	(pointing satellite0 phenomenon8)
	(have_image star4 image2)
	(have_image planet5 thermograph0)
	(have_image star6 spectrograph3)
	(have_image phenomenon7 spectrograph1)
	(have_image phenomenon8 image2)
	(have_image planet9 spectrograph3)
))

)
