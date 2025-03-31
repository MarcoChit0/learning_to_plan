(define (problem prob)
 (:domain barman)
 (:objects 
      shaker447 - shaker
      left right - hand
      shot206 shot406 shot305 - shot
      ingredient153 ingredient463 - ingredient
      cocktail127 - cocktail
      dispenser315 dispenser21 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker447)
  (ontable shot206)
  (ontable shot406)
  (ontable shot305)
  (dispenses dispenser315 ingredient153)
  (dispenses dispenser21 ingredient463)
  (clean shaker447)
  (clean shot206)
  (clean shot406)
  (clean shot305)
  (empty shaker447)
  (empty shot206)
  (empty shot406)
  (empty shot305)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker447 l0)
  (shaker-level shaker447 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail127 ingredient153)
  (cocktail-part2 cocktail127 ingredient463)
)
 (:goal
  (and
      (contains shot206 cocktail127)
      (contains shot406 cocktail127)
)))
