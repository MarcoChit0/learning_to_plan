(define (problem prob)
 (:domain barman)
 (:objects 
      shaker292 - shaker
      left right - hand
      shot275 shot255 - shot
      ingredient104 ingredient279 ingredient414 - ingredient
      cocktail52 - cocktail
      dispenser218 dispenser82 dispenser176 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker292)
  (ontable shot275)
  (ontable shot255)
  (dispenses dispenser218 ingredient104)
  (dispenses dispenser82 ingredient279)
  (dispenses dispenser176 ingredient414)
  (clean shaker292)
  (clean shot275)
  (clean shot255)
  (empty shaker292)
  (empty shot275)
  (empty shot255)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker292 l0)
  (shaker-level shaker292 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail52 ingredient414)
  (cocktail-part2 cocktail52 ingredient104)
)
 (:goal
  (and
      (contains shot275 cocktail52)
)))
