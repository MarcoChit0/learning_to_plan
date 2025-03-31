(define (problem prob)
 (:domain barman)
 (:objects 
      shaker377 - shaker
      left right - hand
      shot372 shot362 - shot
      ingredient321 ingredient80 - ingredient
      cocktail1 - cocktail
      dispenser276 dispenser452 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker377)
  (ontable shot372)
  (ontable shot362)
  (dispenses dispenser276 ingredient321)
  (dispenses dispenser452 ingredient80)
  (clean shaker377)
  (clean shot372)
  (clean shot362)
  (empty shaker377)
  (empty shot372)
  (empty shot362)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker377 l0)
  (shaker-level shaker377 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail1 ingredient321)
  (cocktail-part2 cocktail1 ingredient80)
)
 (:goal
  (and
      (contains shot372 cocktail1)
)))
