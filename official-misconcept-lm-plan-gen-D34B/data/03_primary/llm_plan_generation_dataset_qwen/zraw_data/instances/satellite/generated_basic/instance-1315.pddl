(define (problem strips-sat-x-1)
(:domain satellite)
(:objects
	satellite0 - satellite
	instrument0 - instrument
	instrument1 - instrument
	satellite1 - satellite
	instrument2 - instrument
	instrument3 - instrument
	satellite2 - satellite
	instrument4 - instrument
	instrument5 - instrument
	spectrograph0 - mode
	infrared1 - mode
	thermograph2 - mode
	star1 - direction
	star4 - direction
	groundstation3 - direction
	groundstation0 - direction
	groundstation2 - direction
	groundstation5 - direction
	star6 - direction
	phenomenon7 - direction
	star8 - direction
	phenomenon9 - direction
)
(:init
	(supports instrument0 spectrograph0)
	(calibration_target instrument0 groundstation5)
	(supports instrument1 thermograph2)
	(calibration_target instrument1 groundstation3)
	(calibration_target instrument1 groundstation5)
	(on_board instrument0 satellite0)
	(on_board instrument1 satellite0)
	(power_avail satellite0)
	(pointing satellite0 groundstation5)
	(supports instrument2 thermograph2)
	(supports instrument2 spectrograph0)
	(supports instrument2 infrared1)
	(calibration_target instrument2 groundstation0)
	(supports instrument3 thermograph2)
	(supports instrument3 infrared1)
	(supports instrument3 spectrograph0)
	(calibration_target instrument3 groundstation2)
	(on_board instrument2 satellite1)
	(on_board instrument3 satellite1)
	(power_avail satellite1)
	(pointing satellite1 star6)
	(supports instrument4 thermograph2)
	(calibration_target instrument4 groundstation2)
	(calibration_target instrument4 groundstation0)
	(supports instrument5 thermograph2)
	(supports instrument5 infrared1)
	(calibration_target instrument5 groundstation5)
	(on_board instrument4 satellite2)
	(on_board instrument5 satellite2)
	(power_avail satellite2)
	(pointing satellite2 star4)
)
(:goal (and
	(have_image star6 infrared1)
	(have_image phenomenon7 infrared1)
	(have_image star8 thermograph2)
	(have_image phenomenon9 infrared1)
))

)
