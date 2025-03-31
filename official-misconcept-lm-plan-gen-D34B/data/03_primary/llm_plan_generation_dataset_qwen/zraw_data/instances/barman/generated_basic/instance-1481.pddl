(define (problem prob)
 (:domain barman)
 (:objects 
      shaker406 - shaker
      left right - hand
      shot348 shot439 - shot
      ingredient119 ingredient255 ingredient334 ingredient245 - ingredient
      cocktail139 - cocktail
      dispenser65 dispenser295 dispenser372 dispenser347 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker406)
  (ontable shot348)
  (ontable shot439)
  (dispenses dispenser65 ingredient119)
  (dispenses dispenser295 ingredient255)
  (dispenses dispenser372 ingredient334)
  (dispenses dispenser347 ingredient245)
  (clean shaker406)
  (clean shot348)
  (clean shot439)
  (empty shaker406)
  (empty shot348)
  (empty shot439)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker406 l0)
  (shaker-level shaker406 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail139 ingredient245)
  (cocktail-part2 cocktail139 ingredient334)
)
 (:goal
  (and
      (contains shot348 cocktail139)
)))
