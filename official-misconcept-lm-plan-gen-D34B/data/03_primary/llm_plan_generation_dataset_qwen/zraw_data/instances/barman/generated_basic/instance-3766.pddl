(define (problem prob)
 (:domain barman)
 (:objects 
      shaker17 - shaker
      left right - hand
      shot303 shot192 shot337 - shot
      ingredient237 ingredient301 - ingredient
      cocktail108 - cocktail
      dispenser187 dispenser374 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker17)
  (ontable shot303)
  (ontable shot192)
  (ontable shot337)
  (dispenses dispenser187 ingredient237)
  (dispenses dispenser374 ingredient301)
  (clean shaker17)
  (clean shot303)
  (clean shot192)
  (clean shot337)
  (empty shaker17)
  (empty shot303)
  (empty shot192)
  (empty shot337)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker17 l0)
  (shaker-level shaker17 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail108 ingredient301)
  (cocktail-part2 cocktail108 ingredient237)
)
 (:goal
  (and
      (contains shot303 cocktail108)
      (contains shot192 cocktail108)
)))
