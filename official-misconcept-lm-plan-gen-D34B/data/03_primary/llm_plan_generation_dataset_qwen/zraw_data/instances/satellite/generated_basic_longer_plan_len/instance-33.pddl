(define (problem strips-sat-x-1)
(:domain satellite)
(:objects
	satellite0 - satellite
	instrument0 - instrument
	satellite1 - satellite
	instrument1 - instrument
	instrument2 - instrument
	instrument3 - instrument
	satellite2 - satellite
	instrument4 - instrument
	instrument5 - instrument
	instrument6 - instrument
	instrument7 - instrument
	instrument8 - instrument
	satellite3 - satellite
	instrument9 - instrument
	spectrograph1 - mode
	thermograph0 - mode
	infrared2 - mode
	infrared3 - mode
	star1 - direction
	star3 - direction
	groundstation2 - direction
	star0 - direction
	phenomenon4 - direction
	phenomenon5 - direction
	phenomenon6 - direction
	star7 - direction
	star8 - direction
	phenomenon9 - direction
)
(:init
	(supports instrument0 infrared2)
	(calibration_target instrument0 star3)
	(on_board instrument0 satellite0)
	(power_avail satellite0)
	(pointing satellite0 phenomenon4)
	(supports instrument1 spectrograph1)
	(supports instrument1 infrared3)
	(supports instrument1 thermograph0)
	(calibration_target instrument1 groundstation2)
	(supports instrument2 infrared2)
	(calibration_target instrument2 star3)
	(supports instrument3 spectrograph1)
	(supports instrument3 thermograph0)
	(supports instrument3 infrared3)
	(calibration_target instrument3 groundstation2)
	(on_board instrument1 satellite1)
	(on_board instrument2 satellite1)
	(on_board instrument3 satellite1)
	(power_avail satellite1)
	(pointing satellite1 star0)
	(supports instrument4 thermograph0)
	(supports instrument4 infrared3)
	(calibration_target instrument4 star0)
	(supports instrument5 spectrograph1)
	(supports instrument5 infrared3)
	(calibration_target instrument5 star1)
	(supports instrument6 infrared2)
	(supports instrument6 spectrograph1)
	(supports instrument6 thermograph0)
	(calibration_target instrument6 star0)
	(supports instrument7 infrared2)
	(supports instrument7 infrared3)
	(supports instrument7 spectrograph1)
	(calibration_target instrument7 star3)
	(supports instrument8 infrared3)
	(supports instrument8 infrared2)
	(supports instrument8 spectrograph1)
	(calibration_target instrument8 groundstation2)
	(on_board instrument4 satellite2)
	(on_board instrument5 satellite2)
	(on_board instrument6 satellite2)
	(on_board instrument7 satellite2)
	(on_board instrument8 satellite2)
	(power_avail satellite2)
	(pointing satellite2 phenomenon4)
	(supports instrument9 spectrograph1)
	(supports instrument9 infrared3)
	(calibration_target instrument9 star0)
	(on_board instrument9 satellite3)
	(power_avail satellite3)
	(pointing satellite3 star0)
)
(:goal (and
	(pointing satellite2 star8)
	(have_image phenomenon4 infrared3)
	(have_image phenomenon5 thermograph0)
	(have_image phenomenon6 infrared2)
	(have_image star7 spectrograph1)
	(have_image star8 thermograph0)
	(have_image phenomenon9 thermograph0)
))

)
