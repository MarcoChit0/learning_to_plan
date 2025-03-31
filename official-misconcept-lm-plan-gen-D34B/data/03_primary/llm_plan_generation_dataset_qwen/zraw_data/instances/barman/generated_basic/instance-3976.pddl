(define (problem prob)
 (:domain barman)
 (:objects 
      shaker220 - shaker
      left right - hand
      shot190 shot275 - shot
      ingredient167 ingredient122 ingredient35 ingredient405 - ingredient
      cocktail74 - cocktail
      dispenser91 dispenser68 dispenser99 dispenser82 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker220)
  (ontable shot190)
  (ontable shot275)
  (dispenses dispenser91 ingredient167)
  (dispenses dispenser68 ingredient122)
  (dispenses dispenser99 ingredient35)
  (dispenses dispenser82 ingredient405)
  (clean shaker220)
  (clean shot190)
  (clean shot275)
  (empty shaker220)
  (empty shot190)
  (empty shot275)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker220 l0)
  (shaker-level shaker220 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail74 ingredient122)
  (cocktail-part2 cocktail74 ingredient405)
)
 (:goal
  (and
      (contains shot190 cocktail74)
)))
