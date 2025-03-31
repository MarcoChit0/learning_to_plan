(define (problem prob)
 (:domain barman)
 (:objects 
      shaker57 - shaker
      left right - hand
      shot333 shot475 - shot
      ingredient78 ingredient6 - ingredient
      cocktail1 - cocktail
      dispenser107 dispenser455 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker57)
  (ontable shot333)
  (ontable shot475)
  (dispenses dispenser107 ingredient78)
  (dispenses dispenser455 ingredient6)
  (clean shaker57)
  (clean shot333)
  (clean shot475)
  (empty shaker57)
  (empty shot333)
  (empty shot475)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker57 l0)
  (shaker-level shaker57 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail1 ingredient78)
  (cocktail-part2 cocktail1 ingredient6)
)
 (:goal
  (and
      (contains shot333 cocktail1)
)))
