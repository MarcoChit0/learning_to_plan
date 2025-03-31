(define (problem prob)
 (:domain barman)
 (:objects 
      shaker293 - shaker
      left right - hand
      shot82 - shot
      ingredient9 ingredient87 ingredient453 - ingredient
      cocktail113 - cocktail
      dispenser161 dispenser7 dispenser37 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker293)
  (ontable shot82)
  (dispenses dispenser161 ingredient9)
  (dispenses dispenser7 ingredient87)
  (dispenses dispenser37 ingredient453)
  (clean shaker293)
  (clean shot82)
  (empty shaker293)
  (empty shot82)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker293 l0)
  (shaker-level shaker293 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail113 ingredient87)
  (cocktail-part2 cocktail113 ingredient453)
)
 (:goal
  (and
      (contains shot82 cocktail113)
)))
