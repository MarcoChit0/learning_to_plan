(define (problem prob)
 (:domain barman)
 (:objects 
      shaker338 - shaker
      left right - hand
      shot234 shot24 - shot
      ingredient112 ingredient82 ingredient463 - ingredient
      cocktail1 - cocktail
      dispenser343 dispenser445 dispenser346 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker338)
  (ontable shot234)
  (ontable shot24)
  (dispenses dispenser343 ingredient112)
  (dispenses dispenser445 ingredient82)
  (dispenses dispenser346 ingredient463)
  (clean shaker338)
  (clean shot234)
  (clean shot24)
  (empty shaker338)
  (empty shot234)
  (empty shot24)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker338 l0)
  (shaker-level shaker338 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail1 ingredient463)
  (cocktail-part2 cocktail1 ingredient82)
)
 (:goal
  (and
      (contains shot234 cocktail1)
)))
