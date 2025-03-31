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
	spectrograph3 - mode
	image0 - mode
	spectrograph2 - mode
	thermograph1 - mode
	star1 - direction
	star0 - direction
	star2 - direction
	phenomenon3 - direction
	planet4 - direction
	phenomenon5 - direction
)
(:init
	(supports instrument0 thermograph1)
	(calibration_target instrument0 star1)
	(supports instrument1 spectrograph2)
	(supports instrument1 spectrograph3)
	(calibration_target instrument1 star0)
	(supports instrument2 spectrograph3)
	(calibration_target instrument2 star2)
	(supports instrument3 spectrograph2)
	(supports instrument3 image0)
	(calibration_target instrument3 star0)
	(supports instrument4 thermograph1)
	(supports instrument4 spectrograph3)
	(supports instrument4 spectrograph2)
	(calibration_target instrument4 star2)
	(on_board instrument0 satellite0)
	(on_board instrument1 satellite0)
	(on_board instrument2 satellite0)
	(on_board instrument3 satellite0)
	(on_board instrument4 satellite0)
	(power_avail satellite0)
	(pointing satellite0 phenomenon3)
	(supports instrument5 spectrograph2)
	(supports instrument5 thermograph1)
	(calibration_target instrument5 star0)
	(supports instrument6 image0)
	(supports instrument6 thermograph1)
	(supports instrument6 spectrograph2)
	(calibration_target instrument6 star0)
	(supports instrument7 thermograph1)
	(supports instrument7 image0)
	(calibration_target instrument7 star2)
	(on_board instrument5 satellite1)
	(on_board instrument6 satellite1)
	(on_board instrument7 satellite1)
	(power_avail satellite1)
	(pointing satellite1 planet4)
)
(:goal (and
	(pointing satellite0 planet4)
	(have_image phenomenon3 image0)
	(have_image planet4 spectrograph2)
	(have_image phenomenon5 spectrograph3)
))

)
