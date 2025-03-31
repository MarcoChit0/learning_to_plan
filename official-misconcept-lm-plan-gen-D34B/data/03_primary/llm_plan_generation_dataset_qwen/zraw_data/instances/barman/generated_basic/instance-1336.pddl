(define (problem prob)
 (:domain barman)
 (:objects 
      shaker217 - shaker
      left right - hand
      shot219 shot230 shot178 - shot
      ingredient438 ingredient72 ingredient214 - ingredient
      cocktail375 - cocktail
      dispenser433 dispenser345 dispenser196 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker217)
  (ontable shot219)
  (ontable shot230)
  (ontable shot178)
  (dispenses dispenser433 ingredient438)
  (dispenses dispenser345 ingredient72)
  (dispenses dispenser196 ingredient214)
  (clean shaker217)
  (clean shot219)
  (clean shot230)
  (clean shot178)
  (empty shaker217)
  (empty shot219)
  (empty shot230)
  (empty shot178)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker217 l0)
  (shaker-level shaker217 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail375 ingredient214)
  (cocktail-part2 cocktail375 ingredient438)
)
 (:goal
  (and
      (contains shot219 cocktail375)
      (contains shot230 cocktail375)
)))
