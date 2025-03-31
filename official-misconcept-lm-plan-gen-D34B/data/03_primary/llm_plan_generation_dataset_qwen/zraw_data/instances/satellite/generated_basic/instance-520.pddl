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
	instrument5 - instrument
	thermograph0 - mode
	spectrograph2 - mode
	image3 - mode
	image1 - mode
	groundstation1 - direction
	groundstation2 - direction
	groundstation0 - direction
	groundstation3 - direction
	groundstation4 - direction
	star5 - direction
	phenomenon6 - direction
	planet7 - direction
	star8 - direction
)
(:init
	(supports instrument0 image3)
	(supports instrument0 spectrograph2)
	(supports instrument0 image1)
	(calibration_target instrument0 groundstation0)
	(supports instrument1 spectrograph2)
	(calibration_target instrument1 groundstation4)
	(supports instrument2 thermograph0)
	(supports instrument2 spectrograph2)
	(supports instrument2 image1)
	(calibration_target instrument2 groundstation4)
	(on_board instrument0 satellite0)
	(on_board instrument1 satellite0)
	(on_board instrument2 satellite0)
	(power_avail satellite0)
	(pointing satellite0 groundstation3)
	(supports instrument3 image1)
	(calibration_target instrument3 groundstation3)
	(supports instrument4 image1)
	(supports instrument4 spectrograph2)
	(supports instrument4 thermograph0)
	(calibration_target instrument4 groundstation4)
	(supports instrument5 thermograph0)
	(supports instrument5 image3)
	(calibration_target instrument5 groundstation4)
	(on_board instrument3 satellite1)
	(on_board instrument4 satellite1)
	(on_board instrument5 satellite1)
	(power_avail satellite1)
	(pointing satellite1 groundstation1)
)
(:goal (and
	(pointing satellite0 phenomenon6)
	(have_image star5 image1)
	(have_image phenomenon6 image1)
	(have_image planet7 image1)
	(have_image star8 image1)
))

)
