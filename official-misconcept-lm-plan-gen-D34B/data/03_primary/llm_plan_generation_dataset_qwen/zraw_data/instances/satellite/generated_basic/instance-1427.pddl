(define (problem strips-sat-x-1)
(:domain satellite)
(:objects
	satellite0 - satellite
	instrument0 - instrument
	instrument1 - instrument
	satellite1 - satellite
	instrument2 - instrument
	satellite2 - satellite
	instrument3 - instrument
	instrument4 - instrument
	instrument5 - instrument
	thermograph3 - mode
	infrared2 - mode
	thermograph0 - mode
	spectrograph1 - mode
	groundstation0 - direction
	star1 - direction
	star2 - direction
	phenomenon3 - direction
)
(:init
	(supports instrument0 thermograph3)
	(calibration_target instrument0 groundstation0)
	(supports instrument1 thermograph0)
	(calibration_target instrument1 groundstation0)
	(on_board instrument0 satellite0)
	(on_board instrument1 satellite0)
	(power_avail satellite0)
	(pointing satellite0 phenomenon3)
	(supports instrument2 thermograph3)
	(calibration_target instrument2 groundstation0)
	(on_board instrument2 satellite1)
	(power_avail satellite1)
	(pointing satellite1 groundstation0)
	(supports instrument3 spectrograph1)
	(supports instrument3 thermograph3)
	(supports instrument3 infrared2)
	(calibration_target instrument3 groundstation0)
	(supports instrument4 thermograph3)
	(supports instrument4 spectrograph1)
	(supports instrument4 thermograph0)
	(calibration_target instrument4 groundstation0)
	(supports instrument5 thermograph3)
	(supports instrument5 spectrograph1)
	(calibration_target instrument5 groundstation0)
	(on_board instrument3 satellite2)
	(on_board instrument4 satellite2)
	(on_board instrument5 satellite2)
	(power_avail satellite2)
	(pointing satellite2 star1)
)
(:goal (and
	(pointing satellite0 star1)
	(have_image star1 infrared2)
	(have_image star2 thermograph3)
	(have_image phenomenon3 thermograph0)
))

)
