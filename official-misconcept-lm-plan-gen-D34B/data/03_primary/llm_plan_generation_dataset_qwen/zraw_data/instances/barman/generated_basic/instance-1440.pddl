(define (problem prob)
 (:domain barman)
 (:objects 
      shaker345 - shaker
      left right - hand
      shot350 - shot
      ingredient428 ingredient394 ingredient33 - ingredient
      cocktail356 - cocktail
      dispenser280 dispenser92 dispenser141 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker345)
  (ontable shot350)
  (dispenses dispenser280 ingredient428)
  (dispenses dispenser92 ingredient394)
  (dispenses dispenser141 ingredient33)
  (clean shaker345)
  (clean shot350)
  (empty shaker345)
  (empty shot350)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker345 l0)
  (shaker-level shaker345 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail356 ingredient33)
  (cocktail-part2 cocktail356 ingredient394)
)
 (:goal
  (and
      (contains shot350 cocktail356)
)))
