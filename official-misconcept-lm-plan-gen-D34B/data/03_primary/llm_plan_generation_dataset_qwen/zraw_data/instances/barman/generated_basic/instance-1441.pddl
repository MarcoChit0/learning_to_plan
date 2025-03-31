(define (problem prob)
 (:domain barman)
 (:objects 
      shaker463 - shaker
      left right - hand
      shot225 - shot
      ingredient278 ingredient171 ingredient470 - ingredient
      cocktail243 - cocktail
      dispenser434 dispenser107 dispenser307 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker463)
  (ontable shot225)
  (dispenses dispenser434 ingredient278)
  (dispenses dispenser107 ingredient171)
  (dispenses dispenser307 ingredient470)
  (clean shaker463)
  (clean shot225)
  (empty shaker463)
  (empty shot225)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker463 l0)
  (shaker-level shaker463 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail243 ingredient278)
  (cocktail-part2 cocktail243 ingredient470)
)
 (:goal
  (and
      (contains shot225 cocktail243)
)))
