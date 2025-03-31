(define (problem prob)
 (:domain barman)
 (:objects 
      shaker154 - shaker
      left right - hand
      shot492 - shot
      ingredient366 ingredient403 ingredient93 ingredient438 - ingredient
      cocktail3 - cocktail
      dispenser333 dispenser31 dispenser172 dispenser0 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker154)
  (ontable shot492)
  (dispenses dispenser333 ingredient366)
  (dispenses dispenser31 ingredient403)
  (dispenses dispenser172 ingredient93)
  (dispenses dispenser0 ingredient438)
  (clean shaker154)
  (clean shot492)
  (empty shaker154)
  (empty shot492)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker154 l0)
  (shaker-level shaker154 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail3 ingredient366)
  (cocktail-part2 cocktail3 ingredient93)
)
 (:goal
  (and
      (contains shot492 cocktail3)
)))
