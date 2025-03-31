(define (problem prob)
 (:domain barman)
 (:objects 
      shaker219 - shaker
      left right - hand
      shot122 - shot
      ingredient475 ingredient78 ingredient353 ingredient72 - ingredient
      cocktail56 - cocktail
      dispenser79 dispenser138 dispenser267 dispenser173 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker219)
  (ontable shot122)
  (dispenses dispenser79 ingredient475)
  (dispenses dispenser138 ingredient78)
  (dispenses dispenser267 ingredient353)
  (dispenses dispenser173 ingredient72)
  (clean shaker219)
  (clean shot122)
  (empty shaker219)
  (empty shot122)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker219 l0)
  (shaker-level shaker219 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail56 ingredient353)
  (cocktail-part2 cocktail56 ingredient72)
)
 (:goal
  (and
      (contains shot122 cocktail56)
)))
