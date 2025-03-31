(define (problem strips-sat-x-1)
(:domain satellite)
(:objects
	satellite0 - satellite
	instrument0 - instrument
	instrument1 - instrument
	satellite1 - satellite
	instrument2 - instrument
	instrument3 - instrument
	satellite2 - satellite
	instrument4 - instrument
	image1 - mode
	spectrograph2 - mode
	thermograph0 - mode
	star0 - direction
	star2 - direction
	groundstation5 - direction
	star4 - direction
	star3 - direction
	star1 - direction
	planet6 - direction
	star7 - direction
	phenomenon8 - direction
)
(:init
	(supports instrument0 spectrograph2)
	(calibration_target instrument0 star2)
	(calibration_target instrument0 star3)
	(supports instrument1 image1)
	(calibration_target instrument1 groundstation5)
	(calibration_target instrument1 star2)
	(on_board instrument0 satellite0)
	(on_board instrument1 satellite0)
	(power_avail satellite0)
	(pointing satellite0 star1)
	(supports instrument2 thermograph0)
	(supports instrument2 spectrograph2)
	(supports instrument2 image1)
	(calibration_target instrument2 groundstation5)
	(supports instrument3 image1)
	(supports instrument3 thermograph0)
	(calibration_target instrument3 star3)
	(calibration_target instrument3 star4)
	(on_board instrument2 satellite1)
	(on_board instrument3 satellite1)
	(power_avail satellite1)
	(pointing satellite1 star2)
	(supports instrument4 image1)
	(supports instrument4 thermograph0)
	(calibration_target instrument4 star1)
	(on_board instrument4 satellite2)
	(power_avail satellite2)
	(pointing satellite2 phenomenon8)
)
(:goal (and
	(pointing satellite0 star4)
	(pointing satellite1 groundstation5)
	(have_image planet6 spectrograph2)
	(have_image star7 image1)
	(have_image phenomenon8 image1)
))

)
