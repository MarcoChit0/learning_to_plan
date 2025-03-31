(define (problem prob)
 (:domain barman)
 (:objects 
      shaker277 - shaker
      left right - hand
      shot257 shot259 - shot
      ingredient302 ingredient27 ingredient86 ingredient111 - ingredient
      cocktail326 - cocktail
      dispenser470 dispenser480 dispenser42 dispenser26 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker277)
  (ontable shot257)
  (ontable shot259)
  (dispenses dispenser470 ingredient302)
  (dispenses dispenser480 ingredient27)
  (dispenses dispenser42 ingredient86)
  (dispenses dispenser26 ingredient111)
  (clean shaker277)
  (clean shot257)
  (clean shot259)
  (empty shaker277)
  (empty shot257)
  (empty shot259)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker277 l0)
  (shaker-level shaker277 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail326 ingredient302)
  (cocktail-part2 cocktail326 ingredient86)
)
 (:goal
  (and
      (contains shot257 cocktail326)
)))
