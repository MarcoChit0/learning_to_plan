(define (problem strips-sat-x-1)
(:domain satellite)
(:objects
	satellite0 - satellite
	instrument0 - instrument
	instrument1 - instrument
	satellite1 - satellite
	instrument2 - instrument
	instrument3 - instrument
	instrument4 - instrument
	instrument5 - instrument
	thermograph0 - mode
	thermograph2 - mode
	infrared1 - mode
	spectrograph3 - mode
	groundstation3 - direction
	groundstation0 - direction
	star2 - direction
	star1 - direction
	phenomenon4 - direction
)
(:init
	(supports instrument0 thermograph0)
	(supports instrument0 spectrograph3)
	(calibration_target instrument0 star2)
	(supports instrument1 thermograph2)
	(supports instrument1 infrared1)
	(supports instrument1 spectrograph3)
	(calibration_target instrument1 groundstation0)
	(on_board instrument0 satellite0)
	(on_board instrument1 satellite0)
	(power_avail satellite0)
	(pointing satellite0 groundstation3)
	(supports instrument2 spectrograph3)
	(supports instrument2 thermograph2)
	(calibration_target instrument2 groundstation0)
	(supports instrument3 infrared1)
	(calibration_target instrument3 star2)
	(supports instrument4 thermograph2)
	(supports instrument4 spectrograph3)
	(supports instrument4 thermograph0)
	(calibration_target instrument4 star2)
	(supports instrument5 infrared1)
	(calibration_target instrument5 star1)
	(on_board instrument2 satellite1)
	(on_board instrument3 satellite1)
	(on_board instrument4 satellite1)
	(on_board instrument5 satellite1)
	(power_avail satellite1)
	(pointing satellite1 phenomenon4)
)
(:goal (and
	(pointing satellite1 star2)
	(have_image phenomenon4 thermograph0)
))

)
