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
	satellite2 - satellite
	instrument5 - instrument
	satellite3 - satellite
	instrument6 - instrument
	instrument7 - instrument
	satellite4 - satellite
	instrument8 - instrument
	instrument9 - instrument
	instrument10 - instrument
	instrument11 - instrument
	instrument12 - instrument
	instrument13 - instrument
	thermograph0 - mode
	spectrograph2 - mode
	image1 - mode
	groundstation0 - direction
	planet1 - direction
)
(:init
	(supports instrument0 image1)
	(calibration_target instrument0 groundstation0)
	(on_board instrument0 satellite0)
	(power_avail satellite0)
	(pointing satellite0 groundstation0)
	(supports instrument1 image1)
	(supports instrument1 thermograph0)
	(supports instrument1 spectrograph2)
	(calibration_target instrument1 groundstation0)
	(supports instrument2 spectrograph2)
	(calibration_target instrument2 groundstation0)
	(supports instrument3 thermograph0)
	(supports instrument3 spectrograph2)
	(calibration_target instrument3 groundstation0)
	(supports instrument4 thermograph0)
	(calibration_target instrument4 groundstation0)
	(on_board instrument1 satellite1)
	(on_board instrument2 satellite1)
	(on_board instrument3 satellite1)
	(on_board instrument4 satellite1)
	(power_avail satellite1)
	(pointing satellite1 groundstation0)
	(supports instrument5 thermograph0)
	(calibration_target instrument5 groundstation0)
	(on_board instrument5 satellite2)
	(power_avail satellite2)
	(pointing satellite2 planet1)
	(supports instrument6 image1)
	(calibration_target instrument6 groundstation0)
	(supports instrument7 spectrograph2)
	(calibration_target instrument7 groundstation0)
	(on_board instrument6 satellite3)
	(on_board instrument7 satellite3)
	(power_avail satellite3)
	(pointing satellite3 planet1)
	(supports instrument8 spectrograph2)
	(supports instrument8 thermograph0)
	(supports instrument8 image1)
	(calibration_target instrument8 groundstation0)
	(supports instrument9 thermograph0)
	(supports instrument9 image1)
	(supports instrument9 spectrograph2)
	(calibration_target instrument9 groundstation0)
	(supports instrument10 spectrograph2)
	(calibration_target instrument10 groundstation0)
	(supports instrument11 image1)
	(supports instrument11 spectrograph2)
	(supports instrument11 thermograph0)
	(calibration_target instrument11 groundstation0)
	(supports instrument12 image1)
	(calibration_target instrument12 groundstation0)
	(supports instrument13 image1)
	(supports instrument13 spectrograph2)
	(supports instrument13 thermograph0)
	(calibration_target instrument13 groundstation0)
	(on_board instrument8 satellite4)
	(on_board instrument9 satellite4)
	(on_board instrument10 satellite4)
	(on_board instrument11 satellite4)
	(on_board instrument12 satellite4)
	(on_board instrument13 satellite4)
	(power_avail satellite4)
	(pointing satellite4 groundstation0)
)
(:goal (and
	(pointing satellite3 groundstation0)
	(have_image planet1 image1)
))

)
