(define (problem prob)
 (:domain barman)
 (:objects 
      shaker102 - shaker
      left right - hand
      shot2 shot206 - shot
      ingredient33 ingredient236 - ingredient
      cocktail1 - cocktail
      dispenser4 dispenser429 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker102)
  (ontable shot2)
  (ontable shot206)
  (dispenses dispenser4 ingredient33)
  (dispenses dispenser429 ingredient236)
  (clean shaker102)
  (clean shot2)
  (clean shot206)
  (empty shaker102)
  (empty shot2)
  (empty shot206)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker102 l0)
  (shaker-level shaker102 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail1 ingredient236)
  (cocktail-part2 cocktail1 ingredient33)
)
 (:goal
  (and
      (contains shot2 cocktail1)
)))
