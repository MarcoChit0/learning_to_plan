(define (problem prob)
 (:domain barman)
 (:objects 
      shaker91 - shaker
      left right - hand
      shot352 shot177 shot76 - shot
      ingredient234 ingredient354 ingredient332 ingredient379 - ingredient
      cocktail1 - cocktail
      dispenser171 dispenser261 dispenser486 dispenser391 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker91)
  (ontable shot352)
  (ontable shot177)
  (ontable shot76)
  (dispenses dispenser171 ingredient234)
  (dispenses dispenser261 ingredient354)
  (dispenses dispenser486 ingredient332)
  (dispenses dispenser391 ingredient379)
  (clean shaker91)
  (clean shot352)
  (clean shot177)
  (clean shot76)
  (empty shaker91)
  (empty shot352)
  (empty shot177)
  (empty shot76)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker91 l0)
  (shaker-level shaker91 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail1 ingredient332)
  (cocktail-part2 cocktail1 ingredient234)
)
 (:goal
  (and
      (contains shot352 cocktail1)
      (contains shot177 cocktail1)
)))
