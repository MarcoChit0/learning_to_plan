(define (problem prob)
 (:domain barman)
 (:objects 
      shaker172 - shaker
      left right - hand
      shot146 shot469 shot36 - shot
      ingredient141 ingredient136 ingredient476 - ingredient
      cocktail181 - cocktail
      dispenser319 dispenser258 dispenser457 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker172)
  (ontable shot146)
  (ontable shot469)
  (ontable shot36)
  (dispenses dispenser319 ingredient141)
  (dispenses dispenser258 ingredient136)
  (dispenses dispenser457 ingredient476)
  (clean shaker172)
  (clean shot146)
  (clean shot469)
  (clean shot36)
  (empty shaker172)
  (empty shot146)
  (empty shot469)
  (empty shot36)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker172 l0)
  (shaker-level shaker172 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail181 ingredient141)
  (cocktail-part2 cocktail181 ingredient136)
)
 (:goal
  (and
      (contains shot146 cocktail181)
      (contains shot469 cocktail181)
)))
