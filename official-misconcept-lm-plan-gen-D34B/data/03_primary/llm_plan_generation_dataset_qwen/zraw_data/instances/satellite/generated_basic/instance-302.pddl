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
	image0 - mode
	spectrograph3 - mode
	thermograph2 - mode
	image1 - mode
	star1 - direction
	star2 - direction
	groundstation4 - direction
	star0 - direction
	star5 - direction
	star3 - direction
	phenomenon6 - direction
)
(:init
	(supports instrument0 thermograph2)
	(calibration_target instrument0 star3)
	(calibration_target instrument0 star0)
	(on_board instrument0 satellite0)
	(power_avail satellite0)
	(pointing satellite0 star0)
	(supports instrument1 thermograph2)
	(calibration_target instrument1 star0)
	(on_board instrument1 satellite1)
	(power_avail satellite1)
	(pointing satellite1 star1)
	(supports instrument2 image1)
	(supports instrument2 spectrograph3)
	(supports instrument2 image0)
	(calibration_target instrument2 star5)
	(supports instrument3 thermograph2)
	(calibration_target instrument3 star3)
	(on_board instrument2 satellite2)
	(on_board instrument3 satellite2)
	(power_avail satellite2)
	(pointing satellite2 star0)
)
(:goal (and
	(have_image phenomenon6 thermograph2)
))

)
