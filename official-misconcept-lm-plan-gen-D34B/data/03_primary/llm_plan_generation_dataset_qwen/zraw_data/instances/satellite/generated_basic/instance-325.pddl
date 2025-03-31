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
	thermograph0 - mode
	thermograph1 - mode
	image3 - mode
	spectrograph2 - mode
	groundstation0 - direction
	star2 - direction
	groundstation1 - direction
	groundstation3 - direction
	groundstation5 - direction
	star4 - direction
	phenomenon6 - direction
)
(:init
	(supports instrument0 spectrograph2)
	(supports instrument0 thermograph0)
	(supports instrument0 image3)
	(calibration_target instrument0 groundstation5)
	(on_board instrument0 satellite0)
	(power_avail satellite0)
	(pointing satellite0 star4)
	(supports instrument1 thermograph1)
	(supports instrument1 spectrograph2)
	(calibration_target instrument1 groundstation1)
	(on_board instrument1 satellite1)
	(power_avail satellite1)
	(pointing satellite1 groundstation0)
	(supports instrument2 image3)
	(calibration_target instrument2 groundstation3)
	(calibration_target instrument2 groundstation1)
	(supports instrument3 thermograph1)
	(supports instrument3 thermograph0)
	(supports instrument3 spectrograph2)
	(calibration_target instrument3 star4)
	(supports instrument4 spectrograph2)
	(supports instrument4 thermograph0)
	(calibration_target instrument4 star4)
	(calibration_target instrument4 groundstation5)
	(on_board instrument2 satellite2)
	(on_board instrument3 satellite2)
	(on_board instrument4 satellite2)
	(power_avail satellite2)
	(pointing satellite2 star4)
)
(:goal (and
	(pointing satellite0 phenomenon6)
	(pointing satellite2 groundstation1)
	(have_image phenomenon6 image3)
))

)
