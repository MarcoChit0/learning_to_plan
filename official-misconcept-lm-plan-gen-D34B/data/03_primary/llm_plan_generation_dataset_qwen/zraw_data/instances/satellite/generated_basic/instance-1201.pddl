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
	spectrograph2 - mode
	thermograph1 - mode
	thermograph3 - mode
	groundstation2 - direction
	groundstation0 - direction
	star1 - direction
	phenomenon3 - direction
	phenomenon4 - direction
	star5 - direction
)
(:init
	(supports instrument0 image0)
	(supports instrument0 spectrograph2)
	(calibration_target instrument0 groundstation0)
	(supports instrument1 spectrograph2)
	(supports instrument1 image0)
	(supports instrument1 thermograph1)
	(calibration_target instrument1 star1)
	(supports instrument2 spectrograph2)
	(supports instrument2 thermograph1)
	(supports instrument2 thermograph3)
	(calibration_target instrument2 groundstation2)
	(supports instrument3 spectrograph2)
	(supports instrument3 image0)
	(supports instrument3 thermograph1)
	(calibration_target instrument3 groundstation2)
	(supports instrument4 thermograph3)
	(calibration_target instrument4 groundstation0)
	(on_board instrument0 satellite0)
	(on_board instrument1 satellite0)
	(on_board instrument2 satellite0)
	(on_board instrument3 satellite0)
	(on_board instrument4 satellite0)
	(power_avail satellite0)
	(pointing satellite0 star5)
	(supports instrument5 image0)
	(supports instrument5 spectrograph2)
	(calibration_target instrument5 groundstation0)
	(supports instrument6 spectrograph2)
	(supports instrument6 thermograph3)
	(calibration_target instrument6 groundstation0)
	(supports instrument7 thermograph3)
	(calibration_target instrument7 groundstation0)
	(supports instrument8 thermograph1)
	(calibration_target instrument8 star1)
	(on_board instrument5 satellite1)
	(on_board instrument6 satellite1)
	(on_board instrument7 satellite1)
	(on_board instrument8 satellite1)
	(power_avail satellite1)
	(pointing satellite1 phenomenon3)
)
(:goal (and
	(pointing satellite0 groundstation2)
	(have_image phenomenon3 thermograph3)
	(have_image phenomenon4 thermograph1)
	(have_image star5 image0)
))

)
