(define (problem prob)
 (:domain barman)
 (:objects 
      shaker132 - shaker
      left right - hand
      shot19 shot258 shot62 - shot
      ingredient8 ingredient135 ingredient4 - ingredient
      cocktail328 - cocktail
      dispenser17 dispenser243 dispenser189 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker132)
  (ontable shot19)
  (ontable shot258)
  (ontable shot62)
  (dispenses dispenser17 ingredient8)
  (dispenses dispenser243 ingredient135)
  (dispenses dispenser189 ingredient4)
  (clean shaker132)
  (clean shot19)
  (clean shot258)
  (clean shot62)
  (empty shaker132)
  (empty shot19)
  (empty shot258)
  (empty shot62)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker132 l0)
  (shaker-level shaker132 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail328 ingredient4)
  (cocktail-part2 cocktail328 ingredient135)
)
 (:goal
  (and
      (contains shot19 cocktail328)
      (contains shot258 ingredient4)
)))
