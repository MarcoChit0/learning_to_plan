(define (problem prob)
 (:domain barman)
 (:objects 
      shaker181 - shaker
      left right - hand
      shot123 shot438 - shot
      ingredient398 ingredient494 ingredient184 - ingredient
      cocktail482 - cocktail
      dispenser170 dispenser161 dispenser14 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker181)
  (ontable shot123)
  (ontable shot438)
  (dispenses dispenser170 ingredient398)
  (dispenses dispenser161 ingredient494)
  (dispenses dispenser14 ingredient184)
  (clean shaker181)
  (clean shot123)
  (clean shot438)
  (empty shaker181)
  (empty shot123)
  (empty shot438)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker181 l0)
  (shaker-level shaker181 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail482 ingredient398)
  (cocktail-part2 cocktail482 ingredient184)
)
 (:goal
  (and
      (contains shot123 cocktail482)
)))
