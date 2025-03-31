(define (problem prob)
 (:domain barman)
 (:objects 
      shaker452 - shaker
      left right - hand
      shot140 shot38 shot100 - shot
      ingredient84 ingredient353 - ingredient
      cocktail94 - cocktail
      dispenser290 dispenser65 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker452)
  (ontable shot140)
  (ontable shot38)
  (ontable shot100)
  (dispenses dispenser290 ingredient84)
  (dispenses dispenser65 ingredient353)
  (clean shaker452)
  (clean shot140)
  (clean shot38)
  (clean shot100)
  (empty shaker452)
  (empty shot140)
  (empty shot38)
  (empty shot100)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker452 l0)
  (shaker-level shaker452 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail94 ingredient353)
  (cocktail-part2 cocktail94 ingredient84)
)
 (:goal
  (and
      (contains shot140 cocktail94)
      (contains shot38 cocktail94)
)))
