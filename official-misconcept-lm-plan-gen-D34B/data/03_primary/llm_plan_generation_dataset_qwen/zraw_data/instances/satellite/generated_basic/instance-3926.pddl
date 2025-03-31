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
	satellite2 - satellite
	instrument5 - instrument
	instrument6 - instrument
	instrument7 - instrument
	thermograph1 - mode
	image0 - mode
	spectrograph2 - mode
	star2 - direction
	star0 - direction
	groundstation1 - direction
	planet3 - direction
	phenomenon4 - direction
	planet5 - direction
	star6 - direction
)
(:init
	(supports instrument0 thermograph1)
	(supports instrument0 spectrograph2)
	(calibration_target instrument0 groundstation1)
	(supports instrument1 thermograph1)
	(supports instrument1 spectrograph2)
	(supports instrument1 image0)
	(calibration_target instrument1 groundstation1)
	(on_board instrument0 satellite0)
	(on_board instrument1 satellite0)
	(power_avail satellite0)
	(pointing satellite0 star0)
	(supports instrument2 thermograph1)
	(supports instrument2 spectrograph2)
	(supports instrument2 image0)
	(calibration_target instrument2 groundstation1)
	(supports instrument3 spectrograph2)
	(supports instrument3 image0)
	(supports instrument3 thermograph1)
	(calibration_target instrument3 groundstation1)
	(supports instrument4 image0)
	(supports instrument4 thermograph1)
	(supports instrument4 spectrograph2)
	(calibration_target instrument4 groundstation1)
	(on_board instrument2 satellite1)
	(on_board instrument3 satellite1)
	(on_board instrument4 satellite1)
	(power_avail satellite1)
	(pointing satellite1 groundstation1)
	(supports instrument5 spectrograph2)
	(supports instrument5 thermograph1)
	(supports instrument5 image0)
	(calibration_target instrument5 star2)
	(supports instrument6 thermograph1)
	(supports instrument6 image0)
	(supports instrument6 spectrograph2)
	(calibration_target instrument6 star0)
	(supports instrument7 spectrograph2)
	(supports instrument7 thermograph1)
	(supports instrument7 image0)
	(calibration_target instrument7 groundstation1)
	(on_board instrument5 satellite2)
	(on_board instrument6 satellite2)
	(on_board instrument7 satellite2)
	(power_avail satellite2)
	(pointing satellite2 planet5)
)
(:goal (and
	(pointing satellite1 planet5)
	(have_image planet3 spectrograph2)
	(have_image phenomenon4 thermograph1)
	(have_image planet5 spectrograph2)
	(have_image star6 thermograph1)
))

)
