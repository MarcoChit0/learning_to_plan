(define (problem strips-sat-x-1)
(:domain satellite)
(:objects
	satellite0 - satellite
	instrument0 - instrument
	instrument1 - instrument
	instrument2 - instrument
	satellite1 - satellite
	instrument3 - instrument
	instrument4 - instrument
	image3 - mode
	thermograph1 - mode
	thermograph0 - mode
	spectrograph2 - mode
	star4 - direction
	groundstation3 - direction
	groundstation0 - direction
	star2 - direction
	star1 - direction
	phenomenon5 - direction
	phenomenon6 - direction
	star7 - direction
	planet8 - direction
)
(:init
	(supports instrument0 spectrograph2)
	(supports instrument0 thermograph1)
	(supports instrument0 thermograph0)
	(calibration_target instrument0 groundstation3)
	(supports instrument1 spectrograph2)
	(supports instrument1 image3)
	(calibration_target instrument1 groundstation0)
	(supports instrument2 image3)
	(supports instrument2 thermograph1)
	(calibration_target instrument2 star2)
	(on_board instrument0 satellite0)
	(on_board instrument1 satellite0)
	(on_board instrument2 satellite0)
	(power_avail satellite0)
	(pointing satellite0 star1)
	(supports instrument3 image3)
	(supports instrument3 thermograph0)
	(calibration_target instrument3 star2)
	(supports instrument4 image3)
	(calibration_target instrument4 star1)
	(on_board instrument3 satellite1)
	(on_board instrument4 satellite1)
	(power_avail satellite1)
	(pointing satellite1 star2)
)
(:goal (and
	(have_image phenomenon5 thermograph1)
	(have_image phenomenon6 spectrograph2)
	(have_image star7 spectrograph2)
	(have_image planet8 thermograph1)
))

)
