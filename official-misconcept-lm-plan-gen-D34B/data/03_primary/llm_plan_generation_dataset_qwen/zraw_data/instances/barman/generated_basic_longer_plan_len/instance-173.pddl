(define (problem prob)
 (:domain barman)
 (:objects 
      shaker20 - shaker
      left right - hand
      shot277 shot417 shot149 shot283 - shot
      ingredient300 ingredient347 - ingredient
      cocktail1 - cocktail
      dispenser48 dispenser162 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker20)
  (ontable shot277)
  (ontable shot417)
  (ontable shot149)
  (ontable shot283)
  (dispenses dispenser48 ingredient300)
  (dispenses dispenser162 ingredient347)
  (clean shaker20)
  (clean shot277)
  (clean shot417)
  (clean shot149)
  (clean shot283)
  (empty shaker20)
  (empty shot277)
  (empty shot417)
  (empty shot149)
  (empty shot283)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker20 l0)
  (shaker-level shaker20 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail1 ingredient347)
  (cocktail-part2 cocktail1 ingredient300)
)
 (:goal
  (and
      (contains shot277 cocktail1)
      (contains shot417 ingredient347)
      (contains shot149 ingredient347)
)))
