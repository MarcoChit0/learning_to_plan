(define (problem prob)
 (:domain barman)
 (:objects 
      shaker198 - shaker
      left right - hand
      shot47 shot372 shot204 - shot
      ingredient223 ingredient203 ingredient471 ingredient348 - ingredient
      cocktail36 - cocktail
      dispenser232 dispenser340 dispenser463 dispenser302 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker198)
  (ontable shot47)
  (ontable shot372)
  (ontable shot204)
  (dispenses dispenser232 ingredient223)
  (dispenses dispenser340 ingredient203)
  (dispenses dispenser463 ingredient471)
  (dispenses dispenser302 ingredient348)
  (clean shaker198)
  (clean shot47)
  (clean shot372)
  (clean shot204)
  (empty shaker198)
  (empty shot47)
  (empty shot372)
  (empty shot204)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker198 l0)
  (shaker-level shaker198 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail36 ingredient348)
  (cocktail-part2 cocktail36 ingredient203)
)
 (:goal
  (and
      (contains shot47 cocktail36)
      (contains shot372 cocktail36)
)))
