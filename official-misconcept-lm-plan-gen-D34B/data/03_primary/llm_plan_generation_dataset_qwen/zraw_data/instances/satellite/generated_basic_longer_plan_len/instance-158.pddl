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
	instrument7 - instrument
	instrument8 - instrument
	image3 - mode
	spectrograph1 - mode
	image0 - mode
	thermograph2 - mode
	star1 - direction
	groundstation0 - direction
	groundstation3 - direction
	groundstation4 - direction
	star2 - direction
	planet5 - direction
	phenomenon6 - direction
	phenomenon7 - direction
	planet8 - direction
	star9 - direction
	star10 - direction
	star11 - direction
	star12 - direction
)
(:init
	(supports instrument0 image3)
	(supports instrument0 thermograph2)
	(calibration_target instrument0 groundstation0)
	(supports instrument1 thermograph2)
	(calibration_target instrument1 star1)
	(on_board instrument0 satellite0)
	(on_board instrument1 satellite0)
	(power_avail satellite0)
	(pointing satellite0 star10)
	(supports instrument2 image0)
	(calibration_target instrument2 groundstation0)
	(supports instrument3 spectrograph1)
	(supports instrument3 image3)
	(calibration_target instrument3 groundstation0)
	(supports instrument4 thermograph2)
	(supports instrument4 image3)
	(calibration_target instrument4 groundstation3)
	(supports instrument5 spectrograph1)
	(supports instrument5 image0)
	(supports instrument5 image3)
	(calibration_target instrument5 groundstation4)
	(supports instrument6 image0)
	(supports instrument6 spectrograph1)
	(supports instrument6 image3)
	(calibration_target instrument6 star2)
	(supports instrument7 spectrograph1)
	(supports instrument7 thermograph2)
	(calibration_target instrument7 star2)
	(supports instrument8 spectrograph1)
	(supports instrument8 image0)
	(calibration_target instrument8 star2)
	(on_board instrument2 satellite1)
	(on_board instrument3 satellite1)
	(on_board instrument4 satellite1)
	(on_board instrument5 satellite1)
	(on_board instrument6 satellite1)
	(on_board instrument7 satellite1)
	(on_board instrument8 satellite1)
	(power_avail satellite1)
	(pointing satellite1 planet8)
)
(:goal (and
	(have_image planet5 image0)
	(have_image phenomenon6 image3)
	(have_image phenomenon7 thermograph2)
	(have_image planet8 thermograph2)
	(have_image star9 thermograph2)
	(have_image star10 spectrograph1)
	(have_image star11 spectrograph1)
	(have_image star12 image0)
))

)
