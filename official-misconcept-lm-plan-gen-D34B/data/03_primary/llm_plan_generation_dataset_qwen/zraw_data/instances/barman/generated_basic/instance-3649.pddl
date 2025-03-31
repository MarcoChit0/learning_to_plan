(define (problem prob)
 (:domain barman)
 (:objects 
      shaker234 - shaker
      left right - hand
      shot128 shot478 - shot
      ingredient470 ingredient95 ingredient105 ingredient403 - ingredient
      cocktail230 - cocktail
      dispenser108 dispenser432 dispenser17 dispenser100 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker234)
  (ontable shot128)
  (ontable shot478)
  (dispenses dispenser108 ingredient470)
  (dispenses dispenser432 ingredient95)
  (dispenses dispenser17 ingredient105)
  (dispenses dispenser100 ingredient403)
  (clean shaker234)
  (clean shot128)
  (clean shot478)
  (empty shaker234)
  (empty shot128)
  (empty shot478)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker234 l0)
  (shaker-level shaker234 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail230 ingredient403)
  (cocktail-part2 cocktail230 ingredient95)
)
 (:goal
  (and
      (contains shot128 cocktail230)
)))
