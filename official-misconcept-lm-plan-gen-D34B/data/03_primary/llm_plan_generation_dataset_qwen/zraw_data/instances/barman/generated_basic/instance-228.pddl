(define (problem prob)
 (:domain barman)
 (:objects 
      shaker294 - shaker
      left right - hand
      shot43 - shot
      ingredient125 ingredient217 ingredient167 - ingredient
      cocktail1 - cocktail
      dispenser151 dispenser181 dispenser153 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker294)
  (ontable shot43)
  (dispenses dispenser151 ingredient125)
  (dispenses dispenser181 ingredient217)
  (dispenses dispenser153 ingredient167)
  (clean shaker294)
  (clean shot43)
  (empty shaker294)
  (empty shot43)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker294 l0)
  (shaker-level shaker294 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail1 ingredient167)
  (cocktail-part2 cocktail1 ingredient125)
)
 (:goal
  (and
      (contains shot43 cocktail1)
)))
