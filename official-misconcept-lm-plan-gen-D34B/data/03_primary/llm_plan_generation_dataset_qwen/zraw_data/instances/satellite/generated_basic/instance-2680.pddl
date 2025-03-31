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
	instrument5 - instrument
	spectrograph1 - mode
	image3 - mode
	thermograph2 - mode
	image0 - mode
	star1 - direction
	star3 - direction
	groundstation0 - direction
	star2 - direction
	phenomenon4 - direction
)
(:init
	(supports instrument0 image3)
	(supports instrument0 spectrograph1)
	(supports instrument0 image0)
	(calibration_target instrument0 groundstation0)
	(supports instrument1 image0)
	(calibration_target instrument1 groundstation0)
	(supports instrument2 image0)
	(supports instrument2 spectrograph1)
	(supports instrument2 thermograph2)
	(calibration_target instrument2 star1)
	(supports instrument3 image3)
	(supports instrument3 image0)
	(calibration_target instrument3 star3)
	(on_board instrument0 satellite0)
	(on_board instrument1 satellite0)
	(on_board instrument2 satellite0)
	(on_board instrument3 satellite0)
	(power_avail satellite0)
	(pointing satellite0 star3)
	(supports instrument4 image0)
	(calibration_target instrument4 groundstation0)
	(supports instrument5 image3)
	(supports instrument5 thermograph2)
	(supports instrument5 image0)
	(calibration_target instrument5 star2)
	(on_board instrument4 satellite1)
	(on_board instrument5 satellite1)
	(power_avail satellite1)
	(pointing satellite1 star1)
)
(:goal (and
	(pointing satellite1 phenomenon4)
	(have_image phenomenon4 spectrograph1)
))

)
