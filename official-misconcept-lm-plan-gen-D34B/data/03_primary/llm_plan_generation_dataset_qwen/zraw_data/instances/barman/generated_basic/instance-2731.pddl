(define (problem prob)
 (:domain barman)
 (:objects 
      shaker374 - shaker
      left right - hand
      shot482 shot491 shot78 - shot
      ingredient295 ingredient64 ingredient50 - ingredient
      cocktail22 - cocktail
      dispenser488 dispenser142 dispenser119 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker374)
  (ontable shot482)
  (ontable shot491)
  (ontable shot78)
  (dispenses dispenser488 ingredient295)
  (dispenses dispenser142 ingredient64)
  (dispenses dispenser119 ingredient50)
  (clean shaker374)
  (clean shot482)
  (clean shot491)
  (clean shot78)
  (empty shaker374)
  (empty shot482)
  (empty shot491)
  (empty shot78)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker374 l0)
  (shaker-level shaker374 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail22 ingredient64)
  (cocktail-part2 cocktail22 ingredient50)
)
 (:goal
  (and
      (contains shot482 cocktail22)
      (contains shot491 cocktail22)
)))
