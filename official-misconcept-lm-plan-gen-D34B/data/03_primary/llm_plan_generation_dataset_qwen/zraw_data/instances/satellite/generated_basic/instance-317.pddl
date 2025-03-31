(define (problem strips-sat-x-1)
(:domain satellite)
(:objects
	satellite0 - satellite
	instrument0 - instrument
	instrument1 - instrument
	instrument2 - instrument
	satellite1 - satellite
	instrument3 - instrument
	satellite2 - satellite
	instrument4 - instrument
	instrument5 - instrument
	image2 - mode
	thermograph1 - mode
	spectrograph0 - mode
	image3 - mode
	star1 - direction
	star3 - direction
	groundstation5 - direction
	groundstation0 - direction
	star4 - direction
	groundstation2 - direction
	star6 - direction
)
(:init
	(supports instrument0 image3)
	(calibration_target instrument0 groundstation2)
	(calibration_target instrument0 groundstation0)
	(supports instrument1 thermograph1)
	(supports instrument1 spectrograph0)
	(supports instrument1 image3)
	(calibration_target instrument1 star4)
	(supports instrument2 image2)
	(supports instrument2 image3)
	(supports instrument2 spectrograph0)
	(calibration_target instrument2 groundstation0)
	(on_board instrument0 satellite0)
	(on_board instrument1 satellite0)
	(on_board instrument2 satellite0)
	(power_avail satellite0)
	(pointing satellite0 groundstation5)
	(supports instrument3 thermograph1)
	(supports instrument3 spectrograph0)
	(calibration_target instrument3 groundstation2)
	(calibration_target instrument3 groundstation0)
	(on_board instrument3 satellite1)
	(power_avail satellite1)
	(pointing satellite1 star3)
	(supports instrument4 thermograph1)
	(supports instrument4 image2)
	(calibration_target instrument4 groundstation2)
	(calibration_target instrument4 groundstation0)
	(supports instrument5 image3)
	(calibration_target instrument5 groundstation2)
	(calibration_target instrument5 star4)
	(on_board instrument4 satellite2)
	(on_board instrument5 satellite2)
	(power_avail satellite2)
	(pointing satellite2 star3)
)
(:goal (and
	(pointing satellite0 star4)
	(pointing satellite1 groundstation5)
	(pointing satellite2 star4)
	(have_image star6 image2)
))

)
