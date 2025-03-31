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
	satellite2 - satellite
	instrument5 - instrument
	spectrograph0 - mode
	infrared1 - mode
	image3 - mode
	thermograph2 - mode
	groundstation3 - direction
	groundstation0 - direction
	groundstation5 - direction
	groundstation4 - direction
	star1 - direction
	groundstation2 - direction
	star6 - direction
)
(:init
	(supports instrument0 image3)
	(supports instrument0 spectrograph0)
	(calibration_target instrument0 groundstation3)
	(calibration_target instrument0 groundstation0)
	(supports instrument1 image3)
	(supports instrument1 infrared1)
	(calibration_target instrument1 groundstation0)
	(calibration_target instrument1 groundstation5)
	(on_board instrument0 satellite0)
	(on_board instrument1 satellite0)
	(power_avail satellite0)
	(pointing satellite0 groundstation0)
	(supports instrument2 infrared1)
	(supports instrument2 thermograph2)
	(calibration_target instrument2 groundstation4)
	(calibration_target instrument2 groundstation5)
	(supports instrument3 thermograph2)
	(calibration_target instrument3 groundstation5)
	(calibration_target instrument3 star1)
	(supports instrument4 spectrograph0)
	(supports instrument4 image3)
	(supports instrument4 infrared1)
	(calibration_target instrument4 star1)
	(calibration_target instrument4 groundstation4)
	(on_board instrument2 satellite1)
	(on_board instrument3 satellite1)
	(on_board instrument4 satellite1)
	(power_avail satellite1)
	(pointing satellite1 groundstation5)
	(supports instrument5 spectrograph0)
	(supports instrument5 infrared1)
	(calibration_target instrument5 groundstation2)
	(on_board instrument5 satellite2)
	(power_avail satellite2)
	(pointing satellite2 groundstation5)
)
(:goal (and
	(pointing satellite2 groundstation0)
	(have_image star6 spectrograph0)
))

)
