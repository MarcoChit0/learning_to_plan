(define (problem prob)
 (:domain barman)
 (:objects 
      shaker417 - shaker
      left right - hand
      shot455 shot326 - shot
      ingredient282 ingredient6 ingredient193 - ingredient
      cocktail396 - cocktail
      dispenser304 dispenser36 dispenser368 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker417)
  (ontable shot455)
  (ontable shot326)
  (dispenses dispenser304 ingredient282)
  (dispenses dispenser36 ingredient6)
  (dispenses dispenser368 ingredient193)
  (clean shaker417)
  (clean shot455)
  (clean shot326)
  (empty shaker417)
  (empty shot455)
  (empty shot326)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker417 l0)
  (shaker-level shaker417 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail396 ingredient282)
  (cocktail-part2 cocktail396 ingredient6)
)
 (:goal
  (and
      (contains shot455 cocktail396)
)))
