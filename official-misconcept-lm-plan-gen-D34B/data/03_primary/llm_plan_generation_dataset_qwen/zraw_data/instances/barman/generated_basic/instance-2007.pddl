(define (problem prob)
 (:domain barman)
 (:objects 
      shaker418 - shaker
      left right - hand
      shot340 shot212 shot309 - shot
      ingredient6 ingredient496 ingredient468 - ingredient
      cocktail309 - cocktail
      dispenser157 dispenser400 dispenser161 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker418)
  (ontable shot340)
  (ontable shot212)
  (ontable shot309)
  (dispenses dispenser157 ingredient6)
  (dispenses dispenser400 ingredient496)
  (dispenses dispenser161 ingredient468)
  (clean shaker418)
  (clean shot340)
  (clean shot212)
  (clean shot309)
  (empty shaker418)
  (empty shot340)
  (empty shot212)
  (empty shot309)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker418 l0)
  (shaker-level shaker418 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail309 ingredient6)
  (cocktail-part2 cocktail309 ingredient468)
)
 (:goal
  (and
      (contains shot340 cocktail309)
      (contains shot212 cocktail309)
)))
