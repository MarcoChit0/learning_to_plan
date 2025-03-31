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
	satellite2 - satellite
	instrument6 - instrument
	instrument7 - instrument
	instrument8 - instrument
	thermograph2 - mode
	spectrograph1 - mode
	image0 - mode
	star3 - direction
	star0 - direction
	groundstation2 - direction
	star1 - direction
	phenomenon4 - direction
	phenomenon5 - direction
	planet6 - direction
	star7 - direction
	phenomenon8 - direction
)
(:init
	(supports instrument0 spectrograph1)
	(supports instrument0 thermograph2)
	(supports instrument0 image0)
	(calibration_target instrument0 groundstation2)
	(supports instrument1 spectrograph1)
	(supports instrument1 thermograph2)
	(supports instrument1 image0)
	(calibration_target instrument1 star0)
	(on_board instrument0 satellite0)
	(on_board instrument1 satellite0)
	(power_avail satellite0)
	(pointing satellite0 phenomenon8)
	(supports instrument2 image0)
	(calibration_target instrument2 star1)
	(supports instrument3 thermograph2)
	(calibration_target instrument3 star1)
	(supports instrument4 image0)
	(supports instrument4 thermograph2)
	(calibration_target instrument4 star0)
	(supports instrument5 thermograph2)
	(supports instrument5 spectrograph1)
	(supports instrument5 image0)
	(calibration_target instrument5 star0)
	(on_board instrument2 satellite1)
	(on_board instrument3 satellite1)
	(on_board instrument4 satellite1)
	(on_board instrument5 satellite1)
	(power_avail satellite1)
	(pointing satellite1 planet6)
	(supports instrument6 spectrograph1)
	(calibration_target instrument6 star0)
	(supports instrument7 image0)
	(supports instrument7 spectrograph1)
	(supports instrument7 thermograph2)
	(calibration_target instrument7 groundstation2)
	(supports instrument8 thermograph2)
	(supports instrument8 image0)
	(calibration_target instrument8 star1)
	(on_board instrument6 satellite2)
	(on_board instrument7 satellite2)
	(on_board instrument8 satellite2)
	(power_avail satellite2)
	(pointing satellite2 groundstation2)
)
(:goal (and
	(pointing satellite1 star0)
	(pointing satellite2 star1)
	(have_image phenomenon4 image0)
	(have_image phenomenon5 thermograph2)
	(have_image planet6 spectrograph1)
	(have_image star7 image0)
	(have_image phenomenon8 image0)
))

)
