(define (problem prob)
 (:domain barman)
 (:objects 
      shaker275 - shaker
      left right - hand
      shot280 shot325 - shot
      ingredient353 ingredient63 ingredient84 - ingredient
      cocktail183 - cocktail
      dispenser212 dispenser230 dispenser370 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker275)
  (ontable shot280)
  (ontable shot325)
  (dispenses dispenser212 ingredient353)
  (dispenses dispenser230 ingredient63)
  (dispenses dispenser370 ingredient84)
  (clean shaker275)
  (clean shot280)
  (clean shot325)
  (empty shaker275)
  (empty shot280)
  (empty shot325)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker275 l0)
  (shaker-level shaker275 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail183 ingredient353)
  (cocktail-part2 cocktail183 ingredient63)
)
 (:goal
  (and
      (contains shot280 cocktail183)
)))
