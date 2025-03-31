(define (problem prob)
 (:domain barman)
 (:objects 
      shaker282 - shaker
      left right - hand
      shot424 shot349 - shot
      ingredient128 ingredient198 - ingredient
      cocktail162 - cocktail
      dispenser489 dispenser410 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker282)
  (ontable shot424)
  (ontable shot349)
  (dispenses dispenser489 ingredient128)
  (dispenses dispenser410 ingredient198)
  (clean shaker282)
  (clean shot424)
  (clean shot349)
  (empty shaker282)
  (empty shot424)
  (empty shot349)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker282 l0)
  (shaker-level shaker282 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail162 ingredient128)
  (cocktail-part2 cocktail162 ingredient198)
)
 (:goal
  (and
      (contains shot424 cocktail162)
)))
