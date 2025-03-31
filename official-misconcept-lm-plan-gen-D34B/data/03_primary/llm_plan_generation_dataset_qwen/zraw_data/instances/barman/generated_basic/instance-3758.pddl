(define (problem prob)
 (:domain barman)
 (:objects 
      shaker44 - shaker
      left right - hand
      shot188 shot11 shot330 - shot
      ingredient306 ingredient411 - ingredient
      cocktail32 - cocktail
      dispenser434 dispenser163 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker44)
  (ontable shot188)
  (ontable shot11)
  (ontable shot330)
  (dispenses dispenser434 ingredient306)
  (dispenses dispenser163 ingredient411)
  (clean shaker44)
  (clean shot188)
  (clean shot11)
  (clean shot330)
  (empty shaker44)
  (empty shot188)
  (empty shot11)
  (empty shot330)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker44 l0)
  (shaker-level shaker44 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail32 ingredient411)
  (cocktail-part2 cocktail32 ingredient306)
)
 (:goal
  (and
      (contains shot188 cocktail32)
      (contains shot11 cocktail32)
)))
