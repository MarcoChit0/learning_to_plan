(define (problem strips-sat-x-1)
(:domain satellite)
(:objects
	satellite0 - satellite
	instrument0 - instrument
	instrument1 - instrument
	instrument2 - instrument
	satellite1 - satellite
	instrument3 - instrument
	instrument4 - instrument
	instrument5 - instrument
	instrument6 - instrument
	satellite2 - satellite
	instrument7 - instrument
	instrument8 - instrument
	instrument9 - instrument
	instrument10 - instrument
	image2 - mode
	spectrograph0 - mode
	thermograph1 - mode
	star4 - direction
	star0 - direction
	groundstation3 - direction
	star1 - direction
	groundstation5 - direction
	groundstation2 - direction
	planet6 - direction
	phenomenon7 - direction
	star8 - direction
	phenomenon9 - direction
)
(:init
	(supports instrument0 image2)
	(calibration_target instrument0 groundstation3)
	(supports instrument1 spectrograph0)
	(supports instrument1 image2)
	(supports instrument1 thermograph1)
	(calibration_target instrument1 star4)
	(supports instrument2 image2)
	(supports instrument2 thermograph1)
	(supports instrument2 spectrograph0)
	(calibration_target instrument2 star0)
	(calibration_target instrument2 groundstation3)
	(on_board instrument0 satellite0)
	(on_board instrument1 satellite0)
	(on_board instrument2 satellite0)
	(power_avail satellite0)
	(pointing satellite0 planet6)
	(supports instrument3 image2)
	(supports instrument3 spectrograph0)
	(supports instrument3 thermograph1)
	(calibration_target instrument3 groundstation2)
	(supports instrument4 image2)
	(supports instrument4 thermograph1)
	(supports instrument4 spectrograph0)
	(calibration_target instrument4 groundstation5)
	(supports instrument5 image2)
	(supports instrument5 spectrograph0)
	(calibration_target instrument5 star4)
	(calibration_target instrument5 star1)
	(supports instrument6 thermograph1)
	(supports instrument6 image2)
	(calibration_target instrument6 star0)
	(calibration_target instrument6 groundstation5)
	(on_board instrument3 satellite1)
	(on_board instrument4 satellite1)
	(on_board instrument5 satellite1)
	(on_board instrument6 satellite1)
	(power_avail satellite1)
	(pointing satellite1 planet6)
	(supports instrument7 image2)
	(supports instrument7 thermograph1)
	(calibration_target instrument7 groundstation2)
	(calibration_target instrument7 groundstation3)
	(supports instrument8 image2)
	(supports instrument8 thermograph1)
	(calibration_target instrument8 star1)
	(supports instrument9 spectrograph0)
	(calibration_target instrument9 groundstation5)
	(supports instrument10 image2)
	(supports instrument10 thermograph1)
	(supports instrument10 spectrograph0)
	(calibration_target instrument10 groundstation2)
	(on_board instrument7 satellite2)
	(on_board instrument8 satellite2)
	(on_board instrument9 satellite2)
	(on_board instrument10 satellite2)
	(power_avail satellite2)
	(pointing satellite2 groundstation5)
)
(:goal (and
	(pointing satellite1 groundstation3)
	(have_image planet6 image2)
	(have_image phenomenon7 image2)
	(have_image star8 spectrograph0)
	(have_image phenomenon9 spectrograph0)
))

)
