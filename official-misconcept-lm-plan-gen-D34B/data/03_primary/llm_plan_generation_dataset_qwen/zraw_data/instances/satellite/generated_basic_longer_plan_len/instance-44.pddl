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
	satellite2 - satellite
	instrument7 - instrument
	instrument8 - instrument
	instrument9 - instrument
	satellite3 - satellite
	instrument10 - instrument
	thermograph0 - mode
	spectrograph2 - mode
	thermograph1 - mode
	infrared3 - mode
	groundstation2 - direction
	star3 - direction
	star1 - direction
	star0 - direction
	star4 - direction
	phenomenon5 - direction
	phenomenon6 - direction
	planet7 - direction
	star8 - direction
	phenomenon9 - direction
)
(:init
	(supports instrument0 spectrograph2)
	(supports instrument0 thermograph0)
	(calibration_target instrument0 groundstation2)
	(supports instrument1 infrared3)
	(supports instrument1 spectrograph2)
	(supports instrument1 thermograph1)
	(calibration_target instrument1 star1)
	(supports instrument2 infrared3)
	(supports instrument2 spectrograph2)
	(supports instrument2 thermograph0)
	(calibration_target instrument2 star3)
	(supports instrument3 thermograph1)
	(supports instrument3 thermograph0)
	(supports instrument3 spectrograph2)
	(calibration_target instrument3 groundstation2)
	(supports instrument4 thermograph0)
	(calibration_target instrument4 star0)
	(on_board instrument0 satellite0)
	(on_board instrument1 satellite0)
	(on_board instrument2 satellite0)
	(on_board instrument3 satellite0)
	(on_board instrument4 satellite0)
	(power_avail satellite0)
	(pointing satellite0 phenomenon5)
	(supports instrument5 spectrograph2)
	(supports instrument5 infrared3)
	(calibration_target instrument5 star0)
	(supports instrument6 spectrograph2)
	(supports instrument6 thermograph1)
	(calibration_target instrument6 star3)
	(on_board instrument5 satellite1)
	(on_board instrument6 satellite1)
	(power_avail satellite1)
	(pointing satellite1 groundstation2)
	(supports instrument7 thermograph0)
	(supports instrument7 spectrograph2)
	(supports instrument7 thermograph1)
	(calibration_target instrument7 star1)
	(supports instrument8 thermograph0)
	(supports instrument8 spectrograph2)
	(calibration_target instrument8 star0)
	(supports instrument9 spectrograph2)
	(calibration_target instrument9 star0)
	(on_board instrument7 satellite2)
	(on_board instrument8 satellite2)
	(on_board instrument9 satellite2)
	(power_avail satellite2)
	(pointing satellite2 phenomenon6)
	(supports instrument10 thermograph1)
	(supports instrument10 infrared3)
	(calibration_target instrument10 star0)
	(on_board instrument10 satellite3)
	(power_avail satellite3)
	(pointing satellite3 star8)
)
(:goal (and
	(pointing satellite0 star0)
	(pointing satellite3 phenomenon6)
	(have_image star4 spectrograph2)
	(have_image phenomenon5 thermograph1)
	(have_image phenomenon6 infrared3)
	(have_image planet7 thermograph0)
	(have_image star8 infrared3)
	(have_image phenomenon9 thermograph0)
))

)
