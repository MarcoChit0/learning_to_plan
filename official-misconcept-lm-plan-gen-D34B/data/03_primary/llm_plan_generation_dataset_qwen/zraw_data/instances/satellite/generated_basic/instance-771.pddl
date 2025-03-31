(define (problem strips-sat-x-1)
(:domain satellite)
(:objects
	satellite0 - satellite
	instrument0 - instrument
	satellite1 - satellite
	instrument1 - instrument
	instrument2 - instrument
	instrument3 - instrument
	instrument4 - instrument
	instrument5 - instrument
	instrument6 - instrument
	satellite2 - satellite
	instrument7 - instrument
	satellite3 - satellite
	instrument8 - instrument
	instrument9 - instrument
	instrument10 - instrument
	instrument11 - instrument
	instrument12 - instrument
	image2 - mode
	spectrograph0 - mode
	thermograph1 - mode
	star1 - direction
	star4 - direction
	groundstation2 - direction
	groundstation3 - direction
	star0 - direction
	groundstation5 - direction
	phenomenon6 - direction
	planet7 - direction
	phenomenon8 - direction
	phenomenon9 - direction
	star10 - direction
	phenomenon11 - direction
)
(:init
	(supports instrument0 image2)
	(supports instrument0 thermograph1)
	(calibration_target instrument0 star1)
	(on_board instrument0 satellite0)
	(power_avail satellite0)
	(pointing satellite0 star10)
	(supports instrument1 image2)
	(supports instrument1 thermograph1)
	(supports instrument1 spectrograph0)
	(calibration_target instrument1 star1)
	(supports instrument2 thermograph1)
	(calibration_target instrument2 star0)
	(calibration_target instrument2 groundstation2)
	(supports instrument3 thermograph1)
	(supports instrument3 image2)
	(calibration_target instrument3 star1)
	(supports instrument4 spectrograph0)
	(supports instrument4 thermograph1)
	(calibration_target instrument4 star0)
	(supports instrument5 spectrograph0)
	(supports instrument5 image2)
	(calibration_target instrument5 groundstation2)
	(supports instrument6 thermograph1)
	(calibration_target instrument6 star4)
	(calibration_target instrument6 groundstation3)
	(on_board instrument1 satellite1)
	(on_board instrument2 satellite1)
	(on_board instrument3 satellite1)
	(on_board instrument4 satellite1)
	(on_board instrument5 satellite1)
	(on_board instrument6 satellite1)
	(power_avail satellite1)
	(pointing satellite1 phenomenon6)
	(supports instrument7 spectrograph0)
	(supports instrument7 image2)
	(calibration_target instrument7 star4)
	(on_board instrument7 satellite2)
	(power_avail satellite2)
	(pointing satellite2 phenomenon6)
	(supports instrument8 spectrograph0)
	(supports instrument8 thermograph1)
	(supports instrument8 image2)
	(calibration_target instrument8 groundstation2)
	(supports instrument9 spectrograph0)
	(calibration_target instrument9 groundstation3)
	(supports instrument10 spectrograph0)
	(supports instrument10 image2)
	(calibration_target instrument10 groundstation3)
	(supports instrument11 image2)
	(supports instrument11 thermograph1)
	(supports instrument11 spectrograph0)
	(calibration_target instrument11 star0)
	(supports instrument12 image2)
	(supports instrument12 spectrograph0)
	(supports instrument12 thermograph1)
	(calibration_target instrument12 groundstation5)
	(on_board instrument8 satellite3)
	(on_board instrument9 satellite3)
	(on_board instrument10 satellite3)
	(on_board instrument11 satellite3)
	(on_board instrument12 satellite3)
	(power_avail satellite3)
	(pointing satellite3 star0)
)
(:goal (and
	(have_image phenomenon6 spectrograph0)
	(have_image planet7 spectrograph0)
	(have_image phenomenon8 thermograph1)
	(have_image phenomenon9 image2)
	(have_image star10 spectrograph0)
	(have_image phenomenon11 spectrograph0)
))

)
