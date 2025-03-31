(define (problem prob)
 (:domain barman)
 (:objects 
      shaker163 - shaker
      left right - hand
      shot347 - shot
      ingredient152 ingredient58 ingredient368 - ingredient
      cocktail1 - cocktail
      dispenser243 dispenser155 dispenser382 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker163)
  (ontable shot347)
  (dispenses dispenser243 ingredient152)
  (dispenses dispenser155 ingredient58)
  (dispenses dispenser382 ingredient368)
  (clean shaker163)
  (clean shot347)
  (empty shaker163)
  (empty shot347)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker163 l0)
  (shaker-level shaker163 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail1 ingredient152)
  (cocktail-part2 cocktail1 ingredient58)
)
 (:goal
  (and
      (contains shot347 cocktail1)
)))
