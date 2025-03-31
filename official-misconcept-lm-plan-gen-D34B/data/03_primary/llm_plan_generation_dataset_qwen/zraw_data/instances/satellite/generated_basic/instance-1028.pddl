(define (problem strips-sat-x-1)
(:domain satellite)
(:objects
	satellite0 - satellite
	instrument0 - instrument
	instrument1 - instrument
	instrument2 - instrument
	instrument3 - instrument
	satellite1 - satellite
	instrument4 - instrument
	instrument5 - instrument
	instrument6 - instrument
	satellite2 - satellite
	instrument7 - instrument
	instrument8 - instrument
	instrument9 - instrument
	instrument10 - instrument
	instrument11 - instrument
	spectrograph1 - mode
	thermograph0 - mode
	image2 - mode
	groundstation2 - direction
	groundstation1 - direction
	groundstation3 - direction
	star0 - direction
	planet4 - direction
	planet5 - direction
	planet6 - direction
	phenomenon7 - direction
	star8 - direction
)
(:init
	(supports instrument0 thermograph0)
	(calibration_target instrument0 groundstation2)
	(supports instrument1 spectrograph1)
	(supports instrument1 thermograph0)
	(supports instrument1 image2)
	(calibration_target instrument1 groundstation3)
	(supports instrument2 spectrograph1)
	(supports instrument2 image2)
	(supports instrument2 thermograph0)
	(calibration_target instrument2 star0)
	(supports instrument3 thermograph0)
	(supports instrument3 image2)
	(calibration_target instrument3 groundstation1)
	(on_board instrument0 satellite0)
	(on_board instrument1 satellite0)
	(on_board instrument2 satellite0)
	(on_board instrument3 satellite0)
	(power_avail satellite0)
	(pointing satellite0 groundstation1)
	(supports instrument4 thermograph0)
	(calibration_target instrument4 star0)
	(supports instrument5 image2)
	(calibration_target instrument5 star0)
	(supports instrument6 image2)
	(calibration_target instrument6 groundstation3)
	(on_board instrument4 satellite1)
	(on_board instrument5 satellite1)
	(on_board instrument6 satellite1)
	(power_avail satellite1)
	(pointing satellite1 groundstation2)
	(supports instrument7 image2)
	(supports instrument7 spectrograph1)
	(calibration_target instrument7 groundstation1)
	(supports instrument8 spectrograph1)
	(supports instrument8 image2)
	(supports instrument8 thermograph0)
	(calibration_target instrument8 groundstation3)
	(supports instrument9 spectrograph1)
	(calibration_target instrument9 groundstation3)
	(supports instrument10 thermograph0)
	(calibration_target instrument10 groundstation3)
	(supports instrument11 spectrograph1)
	(supports instrument11 thermograph0)
	(calibration_target instrument11 star0)
	(on_board instrument7 satellite2)
	(on_board instrument8 satellite2)
	(on_board instrument9 satellite2)
	(on_board instrument10 satellite2)
	(on_board instrument11 satellite2)
	(power_avail satellite2)
	(pointing satellite2 planet5)
)
(:goal (and
	(pointing satellite1 planet4)
	(have_image planet4 spectrograph1)
	(have_image planet5 spectrograph1)
	(have_image planet6 spectrograph1)
	(have_image phenomenon7 image2)
	(have_image star8 thermograph0)
))

)
