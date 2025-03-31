(define (problem strips-sat-x-1)
(:domain satellite)
(:objects
	satellite0 - satellite
	instrument0 - instrument
	satellite1 - satellite
	instrument1 - instrument
	instrument2 - instrument
	satellite2 - satellite
	instrument3 - instrument
	instrument4 - instrument
	instrument5 - instrument
	spectrograph2 - mode
	image0 - mode
	image1 - mode
	thermograph3 - mode
	groundstation0 - direction
	star1 - direction
	star2 - direction
	star3 - direction
	phenomenon4 - direction
	star5 - direction
	planet6 - direction
)
(:init
	(supports instrument0 thermograph3)
	(supports instrument0 image0)
	(supports instrument0 image1)
	(calibration_target instrument0 groundstation0)
	(on_board instrument0 satellite0)
	(power_avail satellite0)
	(pointing satellite0 star3)
	(supports instrument1 thermograph3)
	(supports instrument1 image0)
	(calibration_target instrument1 groundstation0)
	(supports instrument2 image0)
	(supports instrument2 spectrograph2)
	(calibration_target instrument2 star1)
	(on_board instrument1 satellite1)
	(on_board instrument2 satellite1)
	(power_avail satellite1)
	(pointing satellite1 star1)
	(supports instrument3 image1)
	(supports instrument3 thermograph3)
	(calibration_target instrument3 star1)
	(supports instrument4 thermograph3)
	(supports instrument4 spectrograph2)
	(supports instrument4 image1)
	(calibration_target instrument4 star1)
	(supports instrument5 spectrograph2)
	(calibration_target instrument5 star1)
	(on_board instrument3 satellite2)
	(on_board instrument4 satellite2)
	(on_board instrument5 satellite2)
	(power_avail satellite2)
	(pointing satellite2 groundstation0)
)
(:goal (and
	(have_image star2 image1)
	(have_image star3 spectrograph2)
	(have_image phenomenon4 spectrograph2)
	(have_image star5 image1)
	(have_image planet6 thermograph3)
))

)
