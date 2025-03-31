(define (problem prob)
 (:domain barman)
 (:objects 
      shaker325 - shaker
      left right - hand
      shot177 shot487 - shot
      ingredient312 ingredient336 ingredient146 - ingredient
      cocktail67 - cocktail
      dispenser16 dispenser53 dispenser254 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker325)
  (ontable shot177)
  (ontable shot487)
  (dispenses dispenser16 ingredient312)
  (dispenses dispenser53 ingredient336)
  (dispenses dispenser254 ingredient146)
  (clean shaker325)
  (clean shot177)
  (clean shot487)
  (empty shaker325)
  (empty shot177)
  (empty shot487)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker325 l0)
  (shaker-level shaker325 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail67 ingredient312)
  (cocktail-part2 cocktail67 ingredient146)
)
 (:goal
  (and
      (contains shot177 cocktail67)
)))
