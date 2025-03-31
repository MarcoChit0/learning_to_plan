(define (problem prob)
 (:domain barman)
 (:objects 
      shaker483 - shaker
      left right - hand
      shot380 shot22 - shot
      ingredient287 ingredient152 - ingredient
      cocktail124 - cocktail
      dispenser151 dispenser470 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker483)
  (ontable shot380)
  (ontable shot22)
  (dispenses dispenser151 ingredient287)
  (dispenses dispenser470 ingredient152)
  (clean shaker483)
  (clean shot380)
  (clean shot22)
  (empty shaker483)
  (empty shot380)
  (empty shot22)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker483 l0)
  (shaker-level shaker483 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail124 ingredient152)
  (cocktail-part2 cocktail124 ingredient287)
)
 (:goal
  (and
      (contains shot380 cocktail124)
)))
