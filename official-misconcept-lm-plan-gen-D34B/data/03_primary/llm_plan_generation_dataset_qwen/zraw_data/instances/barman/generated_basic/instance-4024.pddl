(define (problem prob)
 (:domain barman)
 (:objects 
      shaker38 - shaker
      left right - hand
      shot231 shot198 - shot
      ingredient379 ingredient126 ingredient250 - ingredient
      cocktail136 - cocktail
      dispenser487 dispenser187 dispenser470 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker38)
  (ontable shot231)
  (ontable shot198)
  (dispenses dispenser487 ingredient379)
  (dispenses dispenser187 ingredient126)
  (dispenses dispenser470 ingredient250)
  (clean shaker38)
  (clean shot231)
  (clean shot198)
  (empty shaker38)
  (empty shot231)
  (empty shot198)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker38 l0)
  (shaker-level shaker38 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail136 ingredient250)
  (cocktail-part2 cocktail136 ingredient126)
)
 (:goal
  (and
      (contains shot231 cocktail136)
)))
