(define (problem prob)
 (:domain barman)
 (:objects 
      shaker152 - shaker
      left right - hand
      shot208 shot219 shot366 - shot
      ingredient92 ingredient52 - ingredient
      cocktail242 - cocktail
      dispenser319 dispenser109 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker152)
  (ontable shot208)
  (ontable shot219)
  (ontable shot366)
  (dispenses dispenser319 ingredient92)
  (dispenses dispenser109 ingredient52)
  (clean shaker152)
  (clean shot208)
  (clean shot219)
  (clean shot366)
  (empty shaker152)
  (empty shot208)
  (empty shot219)
  (empty shot366)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker152 l0)
  (shaker-level shaker152 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail242 ingredient52)
  (cocktail-part2 cocktail242 ingredient92)
)
 (:goal
  (and
      (contains shot208 cocktail242)
      (contains shot219 cocktail242)
)))
