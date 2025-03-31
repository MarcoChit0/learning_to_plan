(define (problem prob)
 (:domain barman)
 (:objects 
      shaker192 - shaker
      left right - hand
      shot371 shot274 shot340 - shot
      ingredient189 ingredient160 ingredient200 ingredient350 - ingredient
      cocktail32 - cocktail
      dispenser264 dispenser416 dispenser196 dispenser252 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker192)
  (ontable shot371)
  (ontable shot274)
  (ontable shot340)
  (dispenses dispenser264 ingredient189)
  (dispenses dispenser416 ingredient160)
  (dispenses dispenser196 ingredient200)
  (dispenses dispenser252 ingredient350)
  (clean shaker192)
  (clean shot371)
  (clean shot274)
  (clean shot340)
  (empty shaker192)
  (empty shot371)
  (empty shot274)
  (empty shot340)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker192 l0)
  (shaker-level shaker192 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail32 ingredient160)
  (cocktail-part2 cocktail32 ingredient200)
)
 (:goal
  (and
      (contains shot371 cocktail32)
      (contains shot274 ingredient189)
)))
