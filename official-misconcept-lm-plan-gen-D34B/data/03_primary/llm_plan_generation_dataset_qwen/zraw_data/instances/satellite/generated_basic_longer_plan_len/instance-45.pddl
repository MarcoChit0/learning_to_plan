(define (problem strips-sat-x-1)
(:domain satellite)
(:objects
	satellite0 - satellite
	instrument0 - instrument
	instrument1 - instrument
	instrument2 - instrument
	satellite1 - satellite
	instrument3 - instrument
	satellite2 - satellite
	instrument4 - instrument
	instrument5 - instrument
	instrument6 - instrument
	instrument7 - instrument
	satellite3 - satellite
	instrument8 - instrument
	spectrograph0 - mode
	thermograph1 - mode
	spectrograph3 - mode
	image2 - mode
	groundstation2 - direction
	star0 - direction
	star1 - direction
	groundstation3 - direction
	star4 - direction
	star5 - direction
	phenomenon6 - direction
	star7 - direction
	star8 - direction
	planet9 - direction
)
(:init
	(supports instrument0 spectrograph0)
	(supports instrument0 thermograph1)
	(supports instrument0 image2)
	(calibration_target instrument0 star1)
	(supports instrument1 spectrograph0)
	(supports instrument1 image2)
	(supports instrument1 thermograph1)
	(calibration_target instrument1 star0)
	(supports instrument2 image2)
	(supports instrument2 spectrograph0)
	(supports instrument2 spectrograph3)
	(calibration_target instrument2 star0)
	(on_board instrument0 satellite0)
	(on_board instrument1 satellite0)
	(on_board instrument2 satellite0)
	(power_avail satellite0)
	(pointing satellite0 star5)
	(supports instrument3 image2)
	(supports instrument3 spectrograph3)
	(calibration_target instrument3 star1)
	(on_board instrument3 satellite1)
	(power_avail satellite1)
	(pointing satellite1 groundstation3)
	(supports instrument4 image2)
	(supports instrument4 thermograph1)
	(calibration_target instrument4 star0)
	(supports instrument5 spectrograph0)
	(supports instrument5 image2)
	(supports instrument5 thermograph1)
	(calibration_target instrument5 groundstation2)
	(supports instrument6 thermograph1)
	(supports instrument6 image2)
	(calibration_target instrument6 star0)
	(supports instrument7 spectrograph3)
	(supports instrument7 spectrograph0)
	(supports instrument7 image2)
	(calibration_target instrument7 star1)
	(on_board instrument4 satellite2)
	(on_board instrument5 satellite2)
	(on_board instrument6 satellite2)
	(on_board instrument7 satellite2)
	(power_avail satellite2)
	(pointing satellite2 groundstation3)
	(supports instrument8 image2)
	(supports instrument8 spectrograph3)
	(calibration_target instrument8 groundstation3)
	(on_board instrument8 satellite3)
	(power_avail satellite3)
	(pointing satellite3 star7)
)
(:goal (and
	(pointing satellite2 star5)
	(have_image star4 spectrograph0)
	(have_image star5 image2)
	(have_image phenomenon6 spectrograph0)
	(have_image star7 thermograph1)
	(have_image star8 spectrograph0)
	(have_image planet9 spectrograph0)
))

)
