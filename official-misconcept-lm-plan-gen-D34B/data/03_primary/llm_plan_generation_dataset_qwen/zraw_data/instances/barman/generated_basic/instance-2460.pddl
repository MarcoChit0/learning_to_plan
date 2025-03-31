(define (problem prob)
 (:domain barman)
 (:objects 
      shaker181 - shaker
      left right - hand
      shot116 shot370 shot119 - shot
      ingredient423 ingredient403 ingredient446 ingredient439 - ingredient
      cocktail251 - cocktail
      dispenser386 dispenser336 dispenser83 dispenser117 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker181)
  (ontable shot116)
  (ontable shot370)
  (ontable shot119)
  (dispenses dispenser386 ingredient423)
  (dispenses dispenser336 ingredient403)
  (dispenses dispenser83 ingredient446)
  (dispenses dispenser117 ingredient439)
  (clean shaker181)
  (clean shot116)
  (clean shot370)
  (clean shot119)
  (empty shaker181)
  (empty shot116)
  (empty shot370)
  (empty shot119)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker181 l0)
  (shaker-level shaker181 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail251 ingredient446)
  (cocktail-part2 cocktail251 ingredient403)
)
 (:goal
  (and
      (contains shot116 cocktail251)
      (contains shot370 ingredient439)
)))
