(define (problem prob)
 (:domain barman)
 (:objects 
      shaker22 - shaker
      left right - hand
      shot366 - shot
      ingredient54 ingredient392 ingredient414 - ingredient
      cocktail41 - cocktail
      dispenser298 dispenser99 dispenser11 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker22)
  (ontable shot366)
  (dispenses dispenser298 ingredient54)
  (dispenses dispenser99 ingredient392)
  (dispenses dispenser11 ingredient414)
  (clean shaker22)
  (clean shot366)
  (empty shaker22)
  (empty shot366)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker22 l0)
  (shaker-level shaker22 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail41 ingredient54)
  (cocktail-part2 cocktail41 ingredient392)
)
 (:goal
  (and
      (contains shot366 cocktail41)
)))
