(define (problem prob)
 (:domain barman)
 (:objects 
      shaker489 - shaker
      left right - hand
      shot83 shot154 shot315 - shot
      ingredient363 ingredient207 - ingredient
      cocktail468 - cocktail
      dispenser416 dispenser63 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker489)
  (ontable shot83)
  (ontable shot154)
  (ontable shot315)
  (dispenses dispenser416 ingredient363)
  (dispenses dispenser63 ingredient207)
  (clean shaker489)
  (clean shot83)
  (clean shot154)
  (clean shot315)
  (empty shaker489)
  (empty shot83)
  (empty shot154)
  (empty shot315)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker489 l0)
  (shaker-level shaker489 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail468 ingredient207)
  (cocktail-part2 cocktail468 ingredient363)
)
 (:goal
  (and
      (contains shot83 cocktail468)
      (contains shot154 cocktail468)
)))
