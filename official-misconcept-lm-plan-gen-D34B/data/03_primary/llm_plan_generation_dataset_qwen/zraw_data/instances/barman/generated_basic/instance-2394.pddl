(define (problem prob)
 (:domain barman)
 (:objects 
      shaker192 - shaker
      left right - hand
      shot125 shot475 - shot
      ingredient269 ingredient222 ingredient257 - ingredient
      cocktail132 - cocktail
      dispenser325 dispenser18 dispenser182 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker192)
  (ontable shot125)
  (ontable shot475)
  (dispenses dispenser325 ingredient269)
  (dispenses dispenser18 ingredient222)
  (dispenses dispenser182 ingredient257)
  (clean shaker192)
  (clean shot125)
  (clean shot475)
  (empty shaker192)
  (empty shot125)
  (empty shot475)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker192 l0)
  (shaker-level shaker192 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail132 ingredient269)
  (cocktail-part2 cocktail132 ingredient222)
)
 (:goal
  (and
      (contains shot125 cocktail132)
)))
