(define (problem prob)
 (:domain barman)
 (:objects 
      shaker39 - shaker
      left right - hand
      shot212 shot258 - shot
      ingredient127 ingredient45 ingredient8 - ingredient
      cocktail1 - cocktail
      dispenser378 dispenser433 dispenser217 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker39)
  (ontable shot212)
  (ontable shot258)
  (dispenses dispenser378 ingredient127)
  (dispenses dispenser433 ingredient45)
  (dispenses dispenser217 ingredient8)
  (clean shaker39)
  (clean shot212)
  (clean shot258)
  (empty shaker39)
  (empty shot212)
  (empty shot258)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker39 l0)
  (shaker-level shaker39 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail1 ingredient8)
  (cocktail-part2 cocktail1 ingredient127)
)
 (:goal
  (and
      (contains shot212 cocktail1)
)))
