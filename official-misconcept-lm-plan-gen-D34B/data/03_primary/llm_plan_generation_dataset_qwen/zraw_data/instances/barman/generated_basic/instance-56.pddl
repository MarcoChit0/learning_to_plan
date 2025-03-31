(define (problem prob)
 (:domain barman)
 (:objects 
      shaker124 - shaker
      left right - hand
      shot181 shot0 - shot
      ingredient124 ingredient257 ingredient53 ingredient397 - ingredient
      cocktail1 - cocktail
      dispenser351 dispenser237 dispenser478 dispenser50 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker124)
  (ontable shot181)
  (ontable shot0)
  (dispenses dispenser351 ingredient124)
  (dispenses dispenser237 ingredient257)
  (dispenses dispenser478 ingredient53)
  (dispenses dispenser50 ingredient397)
  (clean shaker124)
  (clean shot181)
  (clean shot0)
  (empty shaker124)
  (empty shot181)
  (empty shot0)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker124 l0)
  (shaker-level shaker124 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail1 ingredient53)
  (cocktail-part2 cocktail1 ingredient124)
)
 (:goal
  (and
      (contains shot181 cocktail1)
)))
