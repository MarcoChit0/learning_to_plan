(define (problem strips-sat-x-1)
(:domain satellite)
(:objects
	satellite0 - satellite
	instrument0 - instrument
	instrument1 - instrument
	satellite1 - satellite
	instrument2 - instrument
	image1 - mode
	spectrograph2 - mode
	thermograph0 - mode
	groundstation0 - direction
	groundstation1 - direction
	groundstation2 - direction
	star3 - direction
	star4 - direction
	star5 - direction
)
(:init
	(supports instrument0 image1)
	(supports instrument0 thermograph0)
	(supports instrument0 spectrograph2)
	(calibration_target instrument0 star3)
	(supports instrument1 thermograph0)
	(calibration_target instrument1 star3)
	(on_board instrument0 satellite0)
	(on_board instrument1 satellite0)
	(power_avail satellite0)
	(pointing satellite0 star3)
	(supports instrument2 thermograph0)
	(supports instrument2 image1)
	(calibration_target instrument2 star4)
	(on_board instrument2 satellite1)
	(power_avail satellite1)
	(pointing satellite1 groundstation1)
)
(:goal (and
	(pointing satellite0 star3)
	(have_image star5 thermograph0)
))

)
