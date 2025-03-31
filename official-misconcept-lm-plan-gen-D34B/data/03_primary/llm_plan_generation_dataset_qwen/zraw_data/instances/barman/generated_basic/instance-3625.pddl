(define (problem prob)
 (:domain barman)
 (:objects 
      shaker99 - shaker
      left right - hand
      shot38 shot475 shot277 - shot
      ingredient295 ingredient203 ingredient169 - ingredient
      cocktail205 - cocktail
      dispenser275 dispenser248 dispenser278 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker99)
  (ontable shot38)
  (ontable shot475)
  (ontable shot277)
  (dispenses dispenser275 ingredient295)
  (dispenses dispenser248 ingredient203)
  (dispenses dispenser278 ingredient169)
  (clean shaker99)
  (clean shot38)
  (clean shot475)
  (clean shot277)
  (empty shaker99)
  (empty shot38)
  (empty shot475)
  (empty shot277)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker99 l0)
  (shaker-level shaker99 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail205 ingredient169)
  (cocktail-part2 cocktail205 ingredient203)
)
 (:goal
  (and
      (contains shot38 cocktail205)
      (contains shot475 ingredient169)
)))
