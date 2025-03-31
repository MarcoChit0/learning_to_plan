(define (problem prob)
 (:domain barman)
 (:objects 
      shaker460 - shaker
      left right - hand
      shot392 - shot
      ingredient47 ingredient12 - ingredient
      cocktail15 - cocktail
      dispenser53 dispenser23 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker460)
  (ontable shot392)
  (dispenses dispenser53 ingredient47)
  (dispenses dispenser23 ingredient12)
  (clean shaker460)
  (clean shot392)
  (empty shaker460)
  (empty shot392)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker460 l0)
  (shaker-level shaker460 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail15 ingredient12)
  (cocktail-part2 cocktail15 ingredient47)
)
 (:goal
  (and
      (contains shot392 cocktail15)
)))
