(define (problem prob)
 (:domain barman)
 (:objects 
      shaker0 - shaker
      left right - hand
      shot320 shot321 - shot
      ingredient317 ingredient329 ingredient391 - ingredient
      cocktail174 - cocktail
      dispenser211 dispenser457 dispenser429 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker0)
  (ontable shot320)
  (ontable shot321)
  (dispenses dispenser211 ingredient317)
  (dispenses dispenser457 ingredient329)
  (dispenses dispenser429 ingredient391)
  (clean shaker0)
  (clean shot320)
  (clean shot321)
  (empty shaker0)
  (empty shot320)
  (empty shot321)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker0 l0)
  (shaker-level shaker0 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail174 ingredient317)
  (cocktail-part2 cocktail174 ingredient329)
)
 (:goal
  (and
      (contains shot320 cocktail174)
)))
