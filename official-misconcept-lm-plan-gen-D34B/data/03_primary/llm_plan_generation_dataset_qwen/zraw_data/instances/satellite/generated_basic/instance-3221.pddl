(define (problem strips-sat-x-1)
(:domain satellite)
(:objects
	satellite0 - satellite
	instrument0 - instrument
	satellite1 - satellite
	instrument1 - instrument
	satellite2 - satellite
	instrument2 - instrument
	instrument3 - instrument
	instrument4 - instrument
	satellite3 - satellite
	instrument5 - instrument
	instrument6 - instrument
	instrument7 - instrument
	instrument8 - instrument
	image0 - mode
	thermograph3 - mode
	infrared2 - mode
	spectrograph1 - mode
	groundstation0 - direction
	planet1 - direction
	planet2 - direction
	planet3 - direction
)
(:init
	(supports instrument0 thermograph3)
	(supports instrument0 infrared2)
	(supports instrument0 image0)
	(calibration_target instrument0 groundstation0)
	(on_board instrument0 satellite0)
	(power_avail satellite0)
	(pointing satellite0 planet3)
	(supports instrument1 infrared2)
	(supports instrument1 image0)
	(supports instrument1 thermograph3)
	(calibration_target instrument1 groundstation0)
	(on_board instrument1 satellite1)
	(power_avail satellite1)
	(pointing satellite1 planet3)
	(supports instrument2 infrared2)
	(supports instrument2 thermograph3)
	(calibration_target instrument2 groundstation0)
	(supports instrument3 image0)
	(calibration_target instrument3 groundstation0)
	(supports instrument4 spectrograph1)
	(supports instrument4 thermograph3)
	(supports instrument4 infrared2)
	(calibration_target instrument4 groundstation0)
	(on_board instrument2 satellite2)
	(on_board instrument3 satellite2)
	(on_board instrument4 satellite2)
	(power_avail satellite2)
	(pointing satellite2 groundstation0)
	(supports instrument5 infrared2)
	(calibration_target instrument5 groundstation0)
	(supports instrument6 image0)
	(supports instrument6 spectrograph1)
	(supports instrument6 thermograph3)
	(calibration_target instrument6 groundstation0)
	(supports instrument7 image0)
	(supports instrument7 infrared2)
	(calibration_target instrument7 groundstation0)
	(supports instrument8 spectrograph1)
	(supports instrument8 thermograph3)
	(supports instrument8 infrared2)
	(calibration_target instrument8 groundstation0)
	(on_board instrument5 satellite3)
	(on_board instrument6 satellite3)
	(on_board instrument7 satellite3)
	(on_board instrument8 satellite3)
	(power_avail satellite3)
	(pointing satellite3 groundstation0)
)
(:goal (and
	(pointing satellite0 planet3)
	(pointing satellite1 planet3)
	(pointing satellite2 groundstation0)
	(have_image planet1 thermograph3)
	(have_image planet2 image0)
	(have_image planet3 infrared2)
))

)
