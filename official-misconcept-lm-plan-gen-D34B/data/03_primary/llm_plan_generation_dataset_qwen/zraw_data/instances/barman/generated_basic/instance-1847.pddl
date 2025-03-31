(define (problem prob)
 (:domain barman)
 (:objects 
      shaker32 - shaker
      left right - hand
      shot207 shot340 - shot
      ingredient101 ingredient482 ingredient474 - ingredient
      cocktail307 - cocktail
      dispenser162 dispenser450 dispenser282 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker32)
  (ontable shot207)
  (ontable shot340)
  (dispenses dispenser162 ingredient101)
  (dispenses dispenser450 ingredient482)
  (dispenses dispenser282 ingredient474)
  (clean shaker32)
  (clean shot207)
  (clean shot340)
  (empty shaker32)
  (empty shot207)
  (empty shot340)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker32 l0)
  (shaker-level shaker32 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail307 ingredient482)
  (cocktail-part2 cocktail307 ingredient101)
)
 (:goal
  (and
      (contains shot207 cocktail307)
)))
