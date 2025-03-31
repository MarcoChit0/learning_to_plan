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
	thermograph0 - mode
	spectrograph2 - mode
	image1 - mode
	image3 - mode
	star0 - direction
	groundstation3 - direction
	star1 - direction
	star2 - direction
	star4 - direction
	star5 - direction
	phenomenon6 - direction
	star7 - direction
	phenomenon8 - direction
	phenomenon9 - direction
	planet10 - direction
	star11 - direction
	star12 - direction
)
(:init
	(supports instrument0 spectrograph2)
	(supports instrument0 thermograph0)
	(supports instrument0 image1)
	(calibration_target instrument0 star0)
	(supports instrument1 spectrograph2)
	(supports instrument1 image1)
	(supports instrument1 thermograph0)
	(calibration_target instrument1 groundstation3)
	(on_board instrument0 satellite0)
	(on_board instrument1 satellite0)
	(power_avail satellite0)
	(pointing satellite0 star2)
	(supports instrument2 image1)
	(supports instrument2 thermograph0)
	(calibration_target instrument2 star1)
	(supports instrument3 image1)
	(supports instrument3 image3)
	(supports instrument3 spectrograph2)
	(calibration_target instrument3 star1)
	(supports instrument4 image3)
	(supports instrument4 spectrograph2)
	(supports instrument4 thermograph0)
	(calibration_target instrument4 star4)
	(supports instrument5 spectrograph2)
	(calibration_target instrument5 star2)
	(supports instrument6 image3)
	(supports instrument6 image1)
	(calibration_target instrument6 star4)
	(supports instrument7 spectrograph2)
	(calibration_target instrument7 star4)
	(supports instrument8 spectrograph2)
	(supports instrument8 thermograph0)
	(calibration_target instrument8 star4)
	(on_board instrument2 satellite1)
	(on_board instrument3 satellite1)
	(on_board instrument4 satellite1)
	(on_board instrument5 satellite1)
	(on_board instrument6 satellite1)
	(on_board instrument7 satellite1)
	(on_board instrument8 satellite1)
	(power_avail satellite1)
	(pointing satellite1 star2)
)
(:goal (and
	(pointing satellite1 phenomenon6)
	(have_image star5 spectrograph2)
	(have_image phenomenon6 spectrograph2)
	(have_image star7 spectrograph2)
	(have_image phenomenon8 image1)
	(have_image phenomenon9 image3)
	(have_image planet10 image1)
	(have_image star11 image3)
	(have_image star12 spectrograph2)
))

)
