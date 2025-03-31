(define (problem strips-sat-x-1)
(:domain satellite)
(:objects
	satellite0 - satellite
	instrument0 - instrument
	instrument1 - instrument
	instrument2 - instrument
	instrument3 - instrument
	instrument4 - instrument
	satellite1 - satellite
	instrument5 - instrument
	thermograph0 - mode
	image3 - mode
	image1 - mode
	spectrograph2 - mode
	star1 - direction
	star0 - direction
	groundstation3 - direction
	groundstation2 - direction
	star4 - direction
	phenomenon5 - direction
	phenomenon6 - direction
	phenomenon7 - direction
	phenomenon8 - direction
	phenomenon9 - direction
	planet10 - direction
	star11 - direction
	planet12 - direction
)
(:init
	(supports instrument0 thermograph0)
	(calibration_target instrument0 star1)
	(supports instrument1 spectrograph2)
	(supports instrument1 image3)
	(supports instrument1 image1)
	(calibration_target instrument1 star0)
	(supports instrument2 thermograph0)
	(supports instrument2 spectrograph2)
	(calibration_target instrument2 groundstation3)
	(supports instrument3 thermograph0)
	(supports instrument3 image3)
	(supports instrument3 spectrograph2)
	(calibration_target instrument3 groundstation2)
	(supports instrument4 thermograph0)
	(supports instrument4 image1)
	(supports instrument4 image3)
	(calibration_target instrument4 groundstation2)
	(on_board instrument0 satellite0)
	(on_board instrument1 satellite0)
	(on_board instrument2 satellite0)
	(on_board instrument3 satellite0)
	(on_board instrument4 satellite0)
	(power_avail satellite0)
	(pointing satellite0 phenomenon9)
	(supports instrument5 image3)
	(supports instrument5 spectrograph2)
	(calibration_target instrument5 star4)
	(on_board instrument5 satellite1)
	(power_avail satellite1)
	(pointing satellite1 star1)
)
(:goal (and
	(pointing satellite1 star4)
	(have_image phenomenon5 spectrograph2)
	(have_image phenomenon6 image3)
	(have_image phenomenon7 image3)
	(have_image phenomenon8 spectrograph2)
	(have_image phenomenon9 thermograph0)
	(have_image planet10 thermograph0)
	(have_image star11 image1)
	(have_image planet12 spectrograph2)
))

)
