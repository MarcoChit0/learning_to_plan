(define (problem prob)
 (:domain barman)
 (:objects 
      shaker222 - shaker
      left right - hand
      shot247 shot210 - shot
      ingredient278 ingredient250 - ingredient
      cocktail459 - cocktail
      dispenser490 dispenser317 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker222)
  (ontable shot247)
  (ontable shot210)
  (dispenses dispenser490 ingredient278)
  (dispenses dispenser317 ingredient250)
  (clean shaker222)
  (clean shot247)
  (clean shot210)
  (empty shaker222)
  (empty shot247)
  (empty shot210)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker222 l0)
  (shaker-level shaker222 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail459 ingredient278)
  (cocktail-part2 cocktail459 ingredient250)
)
 (:goal
  (and
      (contains shot247 cocktail459)
)))
