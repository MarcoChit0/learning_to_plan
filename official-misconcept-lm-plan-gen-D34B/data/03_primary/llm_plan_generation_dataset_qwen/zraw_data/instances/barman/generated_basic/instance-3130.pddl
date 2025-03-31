(define (problem prob)
 (:domain barman)
 (:objects 
      shaker93 - shaker
      left right - hand
      shot98 shot169 shot283 - shot
      ingredient107 ingredient5 - ingredient
      cocktail76 - cocktail
      dispenser271 dispenser485 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker93)
  (ontable shot98)
  (ontable shot169)
  (ontable shot283)
  (dispenses dispenser271 ingredient107)
  (dispenses dispenser485 ingredient5)
  (clean shaker93)
  (clean shot98)
  (clean shot169)
  (clean shot283)
  (empty shaker93)
  (empty shot98)
  (empty shot169)
  (empty shot283)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker93 l0)
  (shaker-level shaker93 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail76 ingredient5)
  (cocktail-part2 cocktail76 ingredient107)
)
 (:goal
  (and
      (contains shot98 cocktail76)
      (contains shot169 cocktail76)
)))
