(define (problem strips-sat-x-1)
(:domain satellite)
(:objects
	satellite0 - satellite
	instrument0 - instrument
	instrument1 - instrument
	satellite1 - satellite
	instrument2 - instrument
	satellite2 - satellite
	instrument3 - instrument
	instrument4 - instrument
	instrument5 - instrument
	instrument6 - instrument
	image2 - mode
	spectrograph3 - mode
	thermograph1 - mode
	image0 - mode
	star1 - direction
	groundstation3 - direction
	star2 - direction
	groundstation0 - direction
	planet4 - direction
	planet5 - direction
	phenomenon6 - direction
)
(:init
	(supports instrument0 spectrograph3)
	(supports instrument0 image0)
	(supports instrument0 image2)
	(calibration_target instrument0 groundstation0)
	(supports instrument1 image0)
	(calibration_target instrument1 groundstation0)
	(on_board instrument0 satellite0)
	(on_board instrument1 satellite0)
	(power_avail satellite0)
	(pointing satellite0 planet5)
	(supports instrument2 spectrograph3)
	(supports instrument2 image0)
	(calibration_target instrument2 groundstation3)
	(on_board instrument2 satellite1)
	(power_avail satellite1)
	(pointing satellite1 groundstation3)
	(supports instrument3 image2)
	(supports instrument3 thermograph1)
	(supports instrument3 image0)
	(calibration_target instrument3 star1)
	(supports instrument4 spectrograph3)
	(calibration_target instrument4 groundstation3)
	(supports instrument5 image0)
	(supports instrument5 thermograph1)
	(calibration_target instrument5 star2)
	(supports instrument6 image0)
	(calibration_target instrument6 groundstation0)
	(on_board instrument3 satellite2)
	(on_board instrument4 satellite2)
	(on_board instrument5 satellite2)
	(on_board instrument6 satellite2)
	(power_avail satellite2)
	(pointing satellite2 planet4)
)
(:goal (and
	(pointing satellite0 groundstation3)
	(pointing satellite1 groundstation0)
	(pointing satellite2 phenomenon6)
	(have_image planet4 image0)
	(have_image planet5 image0)
	(have_image phenomenon6 thermograph1)
))

)
