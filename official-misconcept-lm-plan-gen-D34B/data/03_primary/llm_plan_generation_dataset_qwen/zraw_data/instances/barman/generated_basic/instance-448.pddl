(define (problem prob)
 (:domain barman)
 (:objects 
      shaker52 - shaker
      left right - hand
      shot352 shot166 - shot
      ingredient483 ingredient50 ingredient117 - ingredient
      cocktail1 - cocktail
      dispenser366 dispenser275 dispenser10 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker52)
  (ontable shot352)
  (ontable shot166)
  (dispenses dispenser366 ingredient483)
  (dispenses dispenser275 ingredient50)
  (dispenses dispenser10 ingredient117)
  (clean shaker52)
  (clean shot352)
  (clean shot166)
  (empty shaker52)
  (empty shot352)
  (empty shot166)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker52 l0)
  (shaker-level shaker52 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail1 ingredient483)
  (cocktail-part2 cocktail1 ingredient117)
)
 (:goal
  (and
      (contains shot352 cocktail1)
)))
