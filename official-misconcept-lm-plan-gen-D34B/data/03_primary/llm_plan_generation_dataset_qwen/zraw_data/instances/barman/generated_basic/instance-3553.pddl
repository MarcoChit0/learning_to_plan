(define (problem prob)
 (:domain barman)
 (:objects 
      shaker272 - shaker
      left right - hand
      shot392 - shot
      ingredient355 ingredient26 ingredient247 - ingredient
      cocktail220 - cocktail
      dispenser20 dispenser215 dispenser489 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker272)
  (ontable shot392)
  (dispenses dispenser20 ingredient355)
  (dispenses dispenser215 ingredient26)
  (dispenses dispenser489 ingredient247)
  (clean shaker272)
  (clean shot392)
  (empty shaker272)
  (empty shot392)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker272 l0)
  (shaker-level shaker272 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail220 ingredient26)
  (cocktail-part2 cocktail220 ingredient355)
)
 (:goal
  (and
      (contains shot392 cocktail220)
)))
