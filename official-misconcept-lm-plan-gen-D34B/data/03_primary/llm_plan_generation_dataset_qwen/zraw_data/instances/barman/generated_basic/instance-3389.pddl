(define (problem prob)
 (:domain barman)
 (:objects 
      shaker3 - shaker
      left right - hand
      shot100 shot420 shot300 - shot
      ingredient281 ingredient460 ingredient89 - ingredient
      cocktail356 - cocktail
      dispenser375 dispenser339 dispenser166 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker3)
  (ontable shot100)
  (ontable shot420)
  (ontable shot300)
  (dispenses dispenser375 ingredient281)
  (dispenses dispenser339 ingredient460)
  (dispenses dispenser166 ingredient89)
  (clean shaker3)
  (clean shot100)
  (clean shot420)
  (clean shot300)
  (empty shaker3)
  (empty shot100)
  (empty shot420)
  (empty shot300)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker3 l0)
  (shaker-level shaker3 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail356 ingredient460)
  (cocktail-part2 cocktail356 ingredient281)
)
 (:goal
  (and
      (contains shot100 cocktail356)
      (contains shot420 cocktail356)
)))
