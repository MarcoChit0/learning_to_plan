(define (problem prob)
 (:domain barman)
 (:objects 
      shaker382 - shaker
      left right - hand
      shot174 - shot
      ingredient340 ingredient314 ingredient293 - ingredient
      cocktail1 - cocktail
      dispenser282 dispenser227 dispenser441 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker382)
  (ontable shot174)
  (dispenses dispenser282 ingredient340)
  (dispenses dispenser227 ingredient314)
  (dispenses dispenser441 ingredient293)
  (clean shaker382)
  (clean shot174)
  (empty shaker382)
  (empty shot174)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker382 l0)
  (shaker-level shaker382 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail1 ingredient293)
  (cocktail-part2 cocktail1 ingredient314)
)
 (:goal
  (and
      (contains shot174 cocktail1)
)))
