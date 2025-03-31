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
	satellite2 - satellite
	instrument5 - instrument
	spectrograph0 - mode
	thermograph1 - mode
	spectrograph2 - mode
	star2 - direction
	groundstation5 - direction
	star3 - direction
	star4 - direction
	star1 - direction
	groundstation0 - direction
	planet6 - direction
	phenomenon7 - direction
	phenomenon8 - direction
	phenomenon9 - direction
)
(:init
	(supports instrument0 thermograph1)
	(supports instrument0 spectrograph2)
	(supports instrument0 spectrograph0)
	(calibration_target instrument0 groundstation0)
	(calibration_target instrument0 groundstation5)
	(supports instrument1 thermograph1)
	(calibration_target instrument1 star3)
	(supports instrument2 spectrograph0)
	(supports instrument2 spectrograph2)
	(supports instrument2 thermograph1)
	(calibration_target instrument2 star3)
	(calibration_target instrument2 star1)
	(supports instrument3 thermograph1)
	(supports instrument3 spectrograph0)
	(calibration_target instrument3 star1)
	(on_board instrument0 satellite0)
	(on_board instrument1 satellite0)
	(on_board instrument2 satellite0)
	(on_board instrument3 satellite0)
	(power_avail satellite0)
	(pointing satellite0 star4)
	(supports instrument4 spectrograph2)
	(calibration_target instrument4 star4)
	(on_board instrument4 satellite1)
	(power_avail satellite1)
	(pointing satellite1 phenomenon8)
	(supports instrument5 thermograph1)
	(supports instrument5 spectrograph0)
	(supports instrument5 spectrograph2)
	(calibration_target instrument5 groundstation0)
	(calibration_target instrument5 star1)
	(on_board instrument5 satellite2)
	(power_avail satellite2)
	(pointing satellite2 phenomenon7)
)
(:goal (and
	(pointing satellite1 groundstation0)
	(pointing satellite2 phenomenon8)
	(have_image planet6 thermograph1)
	(have_image phenomenon7 thermograph1)
	(have_image phenomenon8 spectrograph2)
	(have_image phenomenon9 thermograph1)
))

)
