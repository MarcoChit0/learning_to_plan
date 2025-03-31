(define (problem prob)
 (:domain barman)
 (:objects 
      shaker470 - shaker
      left right - hand
      shot366 shot197 - shot
      ingredient234 ingredient354 ingredient108 - ingredient
      cocktail135 - cocktail
      dispenser130 dispenser209 dispenser428 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker470)
  (ontable shot366)
  (ontable shot197)
  (dispenses dispenser130 ingredient234)
  (dispenses dispenser209 ingredient354)
  (dispenses dispenser428 ingredient108)
  (clean shaker470)
  (clean shot366)
  (clean shot197)
  (empty shaker470)
  (empty shot366)
  (empty shot197)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker470 l0)
  (shaker-level shaker470 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail135 ingredient234)
  (cocktail-part2 cocktail135 ingredient108)
)
 (:goal
  (and
      (contains shot366 cocktail135)
)))
