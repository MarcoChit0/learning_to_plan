(define (problem prob)
 (:domain barman)
 (:objects 
      shaker458 - shaker
      left right - hand
      shot266 shot267 - shot
      ingredient353 ingredient447 - ingredient
      cocktail1 - cocktail
      dispenser51 dispenser162 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker458)
  (ontable shot266)
  (ontable shot267)
  (dispenses dispenser51 ingredient353)
  (dispenses dispenser162 ingredient447)
  (clean shaker458)
  (clean shot266)
  (clean shot267)
  (empty shaker458)
  (empty shot266)
  (empty shot267)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker458 l0)
  (shaker-level shaker458 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail1 ingredient353)
  (cocktail-part2 cocktail1 ingredient447)
)
 (:goal
  (and
      (contains shot266 cocktail1)
)))
