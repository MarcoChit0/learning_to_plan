(define (problem prob)
 (:domain barman)
 (:objects 
      shaker56 - shaker
      left right - hand
      shot473 shot327 - shot
      ingredient218 ingredient442 ingredient150 ingredient371 - ingredient
      cocktail423 - cocktail
      dispenser102 dispenser254 dispenser421 dispenser422 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker56)
  (ontable shot473)
  (ontable shot327)
  (dispenses dispenser102 ingredient218)
  (dispenses dispenser254 ingredient442)
  (dispenses dispenser421 ingredient150)
  (dispenses dispenser422 ingredient371)
  (clean shaker56)
  (clean shot473)
  (clean shot327)
  (empty shaker56)
  (empty shot473)
  (empty shot327)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker56 l0)
  (shaker-level shaker56 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail423 ingredient150)
  (cocktail-part2 cocktail423 ingredient371)
)
 (:goal
  (and
      (contains shot473 cocktail423)
)))
