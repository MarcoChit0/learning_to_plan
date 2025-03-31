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
	satellite2 - satellite
	instrument5 - instrument
	instrument6 - instrument
	satellite3 - satellite
	instrument7 - instrument
	image1 - mode
	spectrograph0 - mode
	thermograph2 - mode
	groundstation0 - direction
	groundstation1 - direction
	planet2 - direction
	phenomenon3 - direction
)
(:init
	(supports instrument0 thermograph2)
	(supports instrument0 spectrograph0)
	(calibration_target instrument0 groundstation1)
	(supports instrument1 spectrograph0)
	(calibration_target instrument1 groundstation0)
	(supports instrument2 image1)
	(calibration_target instrument2 groundstation1)
	(supports instrument3 thermograph2)
	(supports instrument3 image1)
	(calibration_target instrument3 groundstation0)
	(on_board instrument0 satellite0)
	(on_board instrument1 satellite0)
	(on_board instrument2 satellite0)
	(on_board instrument3 satellite0)
	(power_avail satellite0)
	(pointing satellite0 phenomenon3)
	(supports instrument4 image1)
	(supports instrument4 thermograph2)
	(supports instrument4 spectrograph0)
	(calibration_target instrument4 groundstation0)
	(on_board instrument4 satellite1)
	(power_avail satellite1)
	(pointing satellite1 phenomenon3)
	(supports instrument5 thermograph2)
	(calibration_target instrument5 groundstation0)
	(supports instrument6 image1)
	(supports instrument6 spectrograph0)
	(calibration_target instrument6 groundstation1)
	(on_board instrument5 satellite2)
	(on_board instrument6 satellite2)
	(power_avail satellite2)
	(pointing satellite2 planet2)
	(supports instrument7 spectrograph0)
	(supports instrument7 image1)
	(supports instrument7 thermograph2)
	(calibration_target instrument7 groundstation1)
	(on_board instrument7 satellite3)
	(power_avail satellite3)
	(pointing satellite3 groundstation1)
)
(:goal (and
	(pointing satellite2 groundstation1)
	(pointing satellite3 planet2)
	(have_image planet2 spectrograph0)
	(have_image phenomenon3 spectrograph0)
))

)
