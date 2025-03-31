(define (problem prob)
 (:domain barman)
 (:objects 
      shaker409 - shaker
      left right - hand
      shot173 - shot
      ingredient138 ingredient452 ingredient51 ingredient337 - ingredient
      cocktail88 - cocktail
      dispenser353 dispenser363 dispenser131 dispenser371 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker409)
  (ontable shot173)
  (dispenses dispenser353 ingredient138)
  (dispenses dispenser363 ingredient452)
  (dispenses dispenser131 ingredient51)
  (dispenses dispenser371 ingredient337)
  (clean shaker409)
  (clean shot173)
  (empty shaker409)
  (empty shot173)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker409 l0)
  (shaker-level shaker409 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail88 ingredient138)
  (cocktail-part2 cocktail88 ingredient337)
)
 (:goal
  (and
      (contains shot173 cocktail88)
)))
