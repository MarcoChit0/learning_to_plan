(define (problem prob)
 (:domain barman)
 (:objects 
      shaker21 - shaker
      left right - hand
      shot370 - shot
      ingredient87 ingredient95 - ingredient
      cocktail323 - cocktail
      dispenser164 dispenser200 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker21)
  (ontable shot370)
  (dispenses dispenser164 ingredient87)
  (dispenses dispenser200 ingredient95)
  (clean shaker21)
  (clean shot370)
  (empty shaker21)
  (empty shot370)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker21 l0)
  (shaker-level shaker21 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail323 ingredient95)
  (cocktail-part2 cocktail323 ingredient87)
)
 (:goal
  (and
      (contains shot370 cocktail323)
)))
