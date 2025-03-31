(define (problem prob)
 (:domain barman)
 (:objects 
      shaker167 - shaker
      left right - hand
      shot241 shot169 - shot
      ingredient80 ingredient359 - ingredient
      cocktail1 - cocktail
      dispenser366 dispenser374 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker167)
  (ontable shot241)
  (ontable shot169)
  (dispenses dispenser366 ingredient80)
  (dispenses dispenser374 ingredient359)
  (clean shaker167)
  (clean shot241)
  (clean shot169)
  (empty shaker167)
  (empty shot241)
  (empty shot169)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker167 l0)
  (shaker-level shaker167 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail1 ingredient80)
  (cocktail-part2 cocktail1 ingredient359)
)
 (:goal
  (and
      (contains shot241 cocktail1)
)))
