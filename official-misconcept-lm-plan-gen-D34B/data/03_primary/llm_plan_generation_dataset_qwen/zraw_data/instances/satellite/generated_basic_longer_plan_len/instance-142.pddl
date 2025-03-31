(define (problem strips-sat-x-1)
(:domain satellite)
(:objects
	satellite0 - satellite
	instrument0 - instrument
	instrument1 - instrument
	instrument2 - instrument
	instrument3 - instrument
	instrument4 - instrument
	instrument5 - instrument
	instrument6 - instrument
	satellite1 - satellite
	instrument7 - instrument
	instrument8 - instrument
	instrument9 - instrument
	thermograph3 - mode
	spectrograph0 - mode
	thermograph1 - mode
	image2 - mode
	star4 - direction
	groundstation0 - direction
	star3 - direction
	groundstation1 - direction
	star2 - direction
	planet5 - direction
	star6 - direction
	phenomenon7 - direction
	star8 - direction
	planet9 - direction
	planet10 - direction
	planet11 - direction
	star12 - direction
)
(:init
	(supports instrument0 spectrograph0)
	(calibration_target instrument0 star4)
	(supports instrument1 thermograph3)
	(supports instrument1 image2)
	(supports instrument1 thermograph1)
	(calibration_target instrument1 groundstation0)
	(supports instrument2 image2)
	(calibration_target instrument2 groundstation1)
	(supports instrument3 thermograph3)
	(calibration_target instrument3 groundstation1)
	(supports instrument4 spectrograph0)
	(calibration_target instrument4 star3)
	(supports instrument5 image2)
	(calibration_target instrument5 star2)
	(supports instrument6 thermograph3)
	(supports instrument6 image2)
	(supports instrument6 spectrograph0)
	(calibration_target instrument6 star3)
	(on_board instrument0 satellite0)
	(on_board instrument1 satellite0)
	(on_board instrument2 satellite0)
	(on_board instrument3 satellite0)
	(on_board instrument4 satellite0)
	(on_board instrument5 satellite0)
	(on_board instrument6 satellite0)
	(power_avail satellite0)
	(pointing satellite0 groundstation0)
	(supports instrument7 image2)
	(calibration_target instrument7 star2)
	(supports instrument8 spectrograph0)
	(supports instrument8 image2)
	(calibration_target instrument8 groundstation1)
	(supports instrument9 image2)
	(supports instrument9 thermograph3)
	(calibration_target instrument9 star2)
	(on_board instrument7 satellite1)
	(on_board instrument8 satellite1)
	(on_board instrument9 satellite1)
	(power_avail satellite1)
	(pointing satellite1 star2)
)
(:goal (and
	(pointing satellite0 star6)
	(have_image planet5 spectrograph0)
	(have_image star6 image2)
	(have_image phenomenon7 image2)
	(have_image star8 spectrograph0)
	(have_image planet9 thermograph1)
	(have_image planet10 thermograph1)
	(have_image planet11 thermograph1)
	(have_image star12 thermograph1)
))

)
