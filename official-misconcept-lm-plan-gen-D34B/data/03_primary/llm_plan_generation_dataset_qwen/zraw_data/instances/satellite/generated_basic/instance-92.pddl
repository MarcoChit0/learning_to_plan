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
	satellite3 - satellite
	instrument7 - instrument
	satellite4 - satellite
	instrument8 - instrument
	instrument9 - instrument
	instrument10 - instrument
	thermograph1 - mode
	spectrograph0 - mode
	groundstation2 - direction
	star1 - direction
	groundstation3 - direction
	star0 - direction
	groundstation5 - direction
	groundstation4 - direction
	phenomenon6 - direction
	planet7 - direction
	star8 - direction
	phenomenon9 - direction
)
(:init
	(supports instrument0 spectrograph0)
	(supports instrument0 thermograph1)
	(calibration_target instrument0 groundstation3)
	(on_board instrument0 satellite0)
	(power_avail satellite0)
	(pointing satellite0 groundstation2)
	(supports instrument1 spectrograph0)
	(supports instrument1 thermograph1)
	(calibration_target instrument1 groundstation2)
	(supports instrument2 thermograph1)
	(supports instrument2 spectrograph0)
	(calibration_target instrument2 groundstation3)
	(calibration_target instrument2 groundstation5)
	(supports instrument3 thermograph1)
	(supports instrument3 spectrograph0)
	(calibration_target instrument3 groundstation5)
	(on_board instrument1 satellite1)
	(on_board instrument2 satellite1)
	(on_board instrument3 satellite1)
	(power_avail satellite1)
	(pointing satellite1 groundstation3)
	(supports instrument4 spectrograph0)
	(calibration_target instrument4 groundstation4)
	(supports instrument5 thermograph1)
	(supports instrument5 spectrograph0)
	(calibration_target instrument5 star0)
	(calibration_target instrument5 groundstation2)
	(supports instrument6 thermograph1)
	(supports instrument6 spectrograph0)
	(calibration_target instrument6 star0)
	(calibration_target instrument6 star1)
	(on_board instrument4 satellite2)
	(on_board instrument5 satellite2)
	(on_board instrument6 satellite2)
	(power_avail satellite2)
	(pointing satellite2 star0)
	(supports instrument7 spectrograph0)
	(supports instrument7 thermograph1)
	(calibration_target instrument7 star0)
	(calibration_target instrument7 groundstation3)
	(on_board instrument7 satellite3)
	(power_avail satellite3)
	(pointing satellite3 groundstation2)
	(supports instrument8 spectrograph0)
	(supports instrument8 thermograph1)
	(calibration_target instrument8 star0)
	(calibration_target instrument8 groundstation4)
	(supports instrument9 spectrograph0)
	(calibration_target instrument9 groundstation5)
	(supports instrument10 spectrograph0)
	(calibration_target instrument10 groundstation4)
	(on_board instrument8 satellite4)
	(on_board instrument9 satellite4)
	(on_board instrument10 satellite4)
	(power_avail satellite4)
	(pointing satellite4 groundstation4)
)
(:goal (and
	(pointing satellite4 phenomenon6)
	(have_image phenomenon6 spectrograph0)
	(have_image planet7 spectrograph0)
	(have_image star8 spectrograph0)
	(have_image phenomenon9 thermograph1)
))

)
