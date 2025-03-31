(define (problem prob)
 (:domain barman)
 (:objects 
      shaker183 - shaker
      left right - hand
      shot164 shot49 - shot
      ingredient207 ingredient128 ingredient406 ingredient116 - ingredient
      cocktail311 - cocktail
      dispenser10 dispenser110 dispenser446 dispenser136 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker183)
  (ontable shot164)
  (ontable shot49)
  (dispenses dispenser10 ingredient207)
  (dispenses dispenser110 ingredient128)
  (dispenses dispenser446 ingredient406)
  (dispenses dispenser136 ingredient116)
  (clean shaker183)
  (clean shot164)
  (clean shot49)
  (empty shaker183)
  (empty shot164)
  (empty shot49)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker183 l0)
  (shaker-level shaker183 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail311 ingredient406)
  (cocktail-part2 cocktail311 ingredient116)
)
 (:goal
  (and
      (contains shot164 cocktail311)
)))
