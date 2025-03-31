(define (problem prob)
 (:domain barman)
 (:objects 
      shaker384 - shaker
      left right - hand
      shot179 shot496 - shot
      ingredient428 ingredient375 ingredient419 - ingredient
      cocktail429 - cocktail
      dispenser0 dispenser378 dispenser440 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker384)
  (ontable shot179)
  (ontable shot496)
  (dispenses dispenser0 ingredient428)
  (dispenses dispenser378 ingredient375)
  (dispenses dispenser440 ingredient419)
  (clean shaker384)
  (clean shot179)
  (clean shot496)
  (empty shaker384)
  (empty shot179)
  (empty shot496)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker384 l0)
  (shaker-level shaker384 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail429 ingredient419)
  (cocktail-part2 cocktail429 ingredient375)
)
 (:goal
  (and
      (contains shot179 cocktail429)
)))
