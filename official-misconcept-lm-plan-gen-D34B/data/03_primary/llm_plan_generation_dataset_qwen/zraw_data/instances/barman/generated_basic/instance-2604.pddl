(define (problem prob)
 (:domain barman)
 (:objects 
      shaker271 - shaker
      left right - hand
      shot149 shot11 - shot
      ingredient440 ingredient366 ingredient333 - ingredient
      cocktail376 - cocktail
      dispenser466 dispenser462 dispenser182 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker271)
  (ontable shot149)
  (ontable shot11)
  (dispenses dispenser466 ingredient440)
  (dispenses dispenser462 ingredient366)
  (dispenses dispenser182 ingredient333)
  (clean shaker271)
  (clean shot149)
  (clean shot11)
  (empty shaker271)
  (empty shot149)
  (empty shot11)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker271 l0)
  (shaker-level shaker271 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail376 ingredient333)
  (cocktail-part2 cocktail376 ingredient440)
)
 (:goal
  (and
      (contains shot149 cocktail376)
)))
