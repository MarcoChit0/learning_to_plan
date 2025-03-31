(define (problem prob)
 (:domain barman)
 (:objects 
      shaker394 - shaker
      left right - hand
      shot117 shot232 shot190 - shot
      ingredient421 ingredient179 - ingredient
      cocktail153 - cocktail
      dispenser158 dispenser415 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker394)
  (ontable shot117)
  (ontable shot232)
  (ontable shot190)
  (dispenses dispenser158 ingredient421)
  (dispenses dispenser415 ingredient179)
  (clean shaker394)
  (clean shot117)
  (clean shot232)
  (clean shot190)
  (empty shaker394)
  (empty shot117)
  (empty shot232)
  (empty shot190)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker394 l0)
  (shaker-level shaker394 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail153 ingredient179)
  (cocktail-part2 cocktail153 ingredient421)
)
 (:goal
  (and
      (contains shot117 cocktail153)
      (contains shot232 cocktail153)
)))
