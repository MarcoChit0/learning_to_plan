(define (problem prob)
 (:domain barman)
 (:objects 
      shaker107 - shaker
      left right - hand
      shot51 - shot
      ingredient286 ingredient96 ingredient183 - ingredient
      cocktail368 - cocktail
      dispenser94 dispenser240 dispenser238 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker107)
  (ontable shot51)
  (dispenses dispenser94 ingredient286)
  (dispenses dispenser240 ingredient96)
  (dispenses dispenser238 ingredient183)
  (clean shaker107)
  (clean shot51)
  (empty shaker107)
  (empty shot51)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker107 l0)
  (shaker-level shaker107 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail368 ingredient96)
  (cocktail-part2 cocktail368 ingredient183)
)
 (:goal
  (and
      (contains shot51 cocktail368)
)))
