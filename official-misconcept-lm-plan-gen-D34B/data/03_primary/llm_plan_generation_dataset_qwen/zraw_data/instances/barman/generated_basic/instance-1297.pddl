(define (problem prob)
 (:domain barman)
 (:objects 
      shaker306 - shaker
      left right - hand
      shot134 shot257 shot7 - shot
      ingredient388 ingredient479 - ingredient
      cocktail485 - cocktail
      dispenser167 dispenser426 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker306)
  (ontable shot134)
  (ontable shot257)
  (ontable shot7)
  (dispenses dispenser167 ingredient388)
  (dispenses dispenser426 ingredient479)
  (clean shaker306)
  (clean shot134)
  (clean shot257)
  (clean shot7)
  (empty shaker306)
  (empty shot134)
  (empty shot257)
  (empty shot7)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker306 l0)
  (shaker-level shaker306 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail485 ingredient388)
  (cocktail-part2 cocktail485 ingredient479)
)
 (:goal
  (and
      (contains shot134 cocktail485)
      (contains shot257 ingredient479)
)))
