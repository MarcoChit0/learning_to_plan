(define (problem prob)
 (:domain barman)
 (:objects 
      shaker338 - shaker
      left right - hand
      shot441 shot166 - shot
      ingredient65 ingredient178 ingredient90 ingredient404 - ingredient
      cocktail82 - cocktail
      dispenser93 dispenser271 dispenser84 dispenser277 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker338)
  (ontable shot441)
  (ontable shot166)
  (dispenses dispenser93 ingredient65)
  (dispenses dispenser271 ingredient178)
  (dispenses dispenser84 ingredient90)
  (dispenses dispenser277 ingredient404)
  (clean shaker338)
  (clean shot441)
  (clean shot166)
  (empty shaker338)
  (empty shot441)
  (empty shot166)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker338 l0)
  (shaker-level shaker338 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail82 ingredient178)
  (cocktail-part2 cocktail82 ingredient65)
)
 (:goal
  (and
      (contains shot441 cocktail82)
)))
