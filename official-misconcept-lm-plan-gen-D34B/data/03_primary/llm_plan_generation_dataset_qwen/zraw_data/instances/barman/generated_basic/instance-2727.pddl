(define (problem prob)
 (:domain barman)
 (:objects 
      shaker375 - shaker
      left right - hand
      shot486 shot262 shot45 - shot
      ingredient329 ingredient183 ingredient99 - ingredient
      cocktail223 - cocktail
      dispenser100 dispenser164 dispenser221 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker375)
  (ontable shot486)
  (ontable shot262)
  (ontable shot45)
  (dispenses dispenser100 ingredient329)
  (dispenses dispenser164 ingredient183)
  (dispenses dispenser221 ingredient99)
  (clean shaker375)
  (clean shot486)
  (clean shot262)
  (clean shot45)
  (empty shaker375)
  (empty shot486)
  (empty shot262)
  (empty shot45)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker375 l0)
  (shaker-level shaker375 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail223 ingredient183)
  (cocktail-part2 cocktail223 ingredient99)
)
 (:goal
  (and
      (contains shot486 cocktail223)
      (contains shot262 cocktail223)
)))
