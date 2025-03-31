(define (problem prob)
 (:domain barman)
 (:objects 
      shaker451 - shaker
      left right - hand
      shot417 shot145 shot271 - shot
      ingredient188 ingredient264 ingredient460 - ingredient
      cocktail450 - cocktail
      dispenser220 dispenser295 dispenser380 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker451)
  (ontable shot417)
  (ontable shot145)
  (ontable shot271)
  (dispenses dispenser220 ingredient188)
  (dispenses dispenser295 ingredient264)
  (dispenses dispenser380 ingredient460)
  (clean shaker451)
  (clean shot417)
  (clean shot145)
  (clean shot271)
  (empty shaker451)
  (empty shot417)
  (empty shot145)
  (empty shot271)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker451 l0)
  (shaker-level shaker451 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail450 ingredient460)
  (cocktail-part2 cocktail450 ingredient188)
)
 (:goal
  (and
      (contains shot417 cocktail450)
      (contains shot145 ingredient460)
)))
