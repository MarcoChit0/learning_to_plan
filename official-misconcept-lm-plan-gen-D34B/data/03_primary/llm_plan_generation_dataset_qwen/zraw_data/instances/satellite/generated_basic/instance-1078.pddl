(define (problem strips-sat-x-1)
(:domain satellite)
(:objects
	satellite0 - satellite
	instrument0 - instrument
	satellite1 - satellite
	instrument1 - instrument
	instrument2 - instrument
	instrument3 - instrument
	satellite2 - satellite
	instrument4 - instrument
	instrument5 - instrument
	image2 - mode
	spectrograph0 - mode
	thermograph1 - mode
	star0 - direction
	star5 - direction
	groundstation3 - direction
	star1 - direction
	star2 - direction
	star4 - direction
	planet6 - direction
	star7 - direction
	phenomenon8 - direction
)
(:init
	(supports instrument0 thermograph1)
	(supports instrument0 spectrograph0)
	(calibration_target instrument0 star4)
	(calibration_target instrument0 star0)
	(on_board instrument0 satellite0)
	(power_avail satellite0)
	(pointing satellite0 phenomenon8)
	(supports instrument1 thermograph1)
	(supports instrument1 image2)
	(calibration_target instrument1 groundstation3)
	(calibration_target instrument1 star1)
	(supports instrument2 spectrograph0)
	(supports instrument2 thermograph1)
	(calibration_target instrument2 star5)
	(supports instrument3 thermograph1)
	(supports instrument3 spectrograph0)
	(supports instrument3 image2)
	(calibration_target instrument3 groundstation3)
	(on_board instrument1 satellite1)
	(on_board instrument2 satellite1)
	(on_board instrument3 satellite1)
	(power_avail satellite1)
	(pointing satellite1 star5)
	(supports instrument4 spectrograph0)
	(supports instrument4 image2)
	(calibration_target instrument4 star2)
	(calibration_target instrument4 star1)
	(supports instrument5 spectrograph0)
	(supports instrument5 image2)
	(supports instrument5 thermograph1)
	(calibration_target instrument5 star4)
	(on_board instrument4 satellite2)
	(on_board instrument5 satellite2)
	(power_avail satellite2)
	(pointing satellite2 star4)
)
(:goal (and
	(have_image planet6 image2)
	(have_image star7 thermograph1)
	(have_image phenomenon8 spectrograph0)
))

)
