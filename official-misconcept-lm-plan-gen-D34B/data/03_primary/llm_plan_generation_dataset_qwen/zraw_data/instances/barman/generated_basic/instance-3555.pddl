(define (problem prob)
 (:domain barman)
 (:objects 
      shaker261 - shaker
      left right - hand
      shot317 shot360 - shot
      ingredient4 ingredient188 ingredient85 ingredient395 - ingredient
      cocktail155 - cocktail
      dispenser192 dispenser111 dispenser64 dispenser68 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker261)
  (ontable shot317)
  (ontable shot360)
  (dispenses dispenser192 ingredient4)
  (dispenses dispenser111 ingredient188)
  (dispenses dispenser64 ingredient85)
  (dispenses dispenser68 ingredient395)
  (clean shaker261)
  (clean shot317)
  (clean shot360)
  (empty shaker261)
  (empty shot317)
  (empty shot360)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker261 l0)
  (shaker-level shaker261 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail155 ingredient4)
  (cocktail-part2 cocktail155 ingredient395)
)
 (:goal
  (and
      (contains shot317 cocktail155)
)))
