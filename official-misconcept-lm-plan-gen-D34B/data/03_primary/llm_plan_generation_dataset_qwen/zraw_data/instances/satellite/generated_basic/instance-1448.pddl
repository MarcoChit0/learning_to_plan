(define (problem strips-sat-x-1)
(:domain satellite)
(:objects
	satellite0 - satellite
	instrument0 - instrument
	instrument1 - instrument
	instrument2 - instrument
	satellite1 - satellite
	instrument3 - instrument
	satellite2 - satellite
	instrument4 - instrument
	thermograph3 - mode
	thermograph1 - mode
	image0 - mode
	spectrograph2 - mode
	star0 - direction
	planet1 - direction
	phenomenon2 - direction
	planet3 - direction
)
(:init
	(supports instrument0 spectrograph2)
	(calibration_target instrument0 star0)
	(supports instrument1 thermograph1)
	(calibration_target instrument1 star0)
	(supports instrument2 image0)
	(supports instrument2 thermograph3)
	(calibration_target instrument2 star0)
	(on_board instrument0 satellite0)
	(on_board instrument1 satellite0)
	(on_board instrument2 satellite0)
	(power_avail satellite0)
	(pointing satellite0 phenomenon2)
	(supports instrument3 thermograph1)
	(supports instrument3 image0)
	(calibration_target instrument3 star0)
	(on_board instrument3 satellite1)
	(power_avail satellite1)
	(pointing satellite1 planet3)
	(supports instrument4 thermograph3)
	(calibration_target instrument4 star0)
	(on_board instrument4 satellite2)
	(power_avail satellite2)
	(pointing satellite2 planet3)
)
(:goal (and
	(pointing satellite2 planet1)
	(have_image planet1 thermograph3)
	(have_image phenomenon2 spectrograph2)
	(have_image planet3 image0)
))

)
