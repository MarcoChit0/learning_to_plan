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
	instrument6 - instrument
	infrared3 - mode
	thermograph0 - mode
	spectrograph1 - mode
	image2 - mode
	groundstation2 - direction
	star3 - direction
	groundstation4 - direction
	groundstation1 - direction
	groundstation5 - direction
	groundstation0 - direction
	phenomenon6 - direction
)
(:init
	(supports instrument0 infrared3)
	(supports instrument0 thermograph0)
	(supports instrument0 spectrograph1)
	(calibration_target instrument0 groundstation2)
	(calibration_target instrument0 groundstation1)
	(supports instrument1 infrared3)
	(supports instrument1 thermograph0)
	(supports instrument1 spectrograph1)
	(calibration_target instrument1 groundstation1)
	(calibration_target instrument1 groundstation2)
	(on_board instrument0 satellite0)
	(on_board instrument1 satellite0)
	(power_avail satellite0)
	(pointing satellite0 groundstation5)
	(supports instrument2 thermograph0)
	(calibration_target instrument2 groundstation0)
	(supports instrument3 thermograph0)
	(supports instrument3 spectrograph1)
	(calibration_target instrument3 groundstation0)
	(calibration_target instrument3 star3)
	(on_board instrument2 satellite1)
	(on_board instrument3 satellite1)
	(power_avail satellite1)
	(pointing satellite1 groundstation1)
	(supports instrument4 spectrograph1)
	(supports instrument4 thermograph0)
	(calibration_target instrument4 groundstation5)
	(calibration_target instrument4 groundstation0)
	(supports instrument5 thermograph0)
	(calibration_target instrument5 groundstation1)
	(calibration_target instrument5 groundstation4)
	(supports instrument6 image2)
	(supports instrument6 spectrograph1)
	(supports instrument6 infrared3)
	(calibration_target instrument6 groundstation0)
	(calibration_target instrument6 groundstation5)
	(on_board instrument4 satellite2)
	(on_board instrument5 satellite2)
	(on_board instrument6 satellite2)
	(power_avail satellite2)
	(pointing satellite2 star3)
)
(:goal (and
	(pointing satellite1 groundstation2)
	(have_image phenomenon6 infrared3)
))

)
