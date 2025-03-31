(define (problem prob)
 (:domain barman)
 (:objects 
      shaker21 - shaker
      left right - hand
      shot388 shot199 - shot
      ingredient349 ingredient381 ingredient485 ingredient466 - ingredient
      cocktail1 - cocktail
      dispenser26 dispenser51 dispenser79 dispenser292 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker21)
  (ontable shot388)
  (ontable shot199)
  (dispenses dispenser26 ingredient349)
  (dispenses dispenser51 ingredient381)
  (dispenses dispenser79 ingredient485)
  (dispenses dispenser292 ingredient466)
  (clean shaker21)
  (clean shot388)
  (clean shot199)
  (empty shaker21)
  (empty shot388)
  (empty shot199)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker21 l0)
  (shaker-level shaker21 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail1 ingredient349)
  (cocktail-part2 cocktail1 ingredient381)
)
 (:goal
  (and
      (contains shot388 cocktail1)
)))
