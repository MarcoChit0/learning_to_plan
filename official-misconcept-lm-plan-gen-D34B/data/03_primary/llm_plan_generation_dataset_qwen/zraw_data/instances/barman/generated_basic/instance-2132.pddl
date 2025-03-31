(define (problem prob)
 (:domain barman)
 (:objects 
      shaker113 - shaker
      left right - hand
      shot244 shot122 shot474 - shot
      ingredient289 ingredient116 - ingredient
      cocktail218 - cocktail
      dispenser274 dispenser204 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker113)
  (ontable shot244)
  (ontable shot122)
  (ontable shot474)
  (dispenses dispenser274 ingredient289)
  (dispenses dispenser204 ingredient116)
  (clean shaker113)
  (clean shot244)
  (clean shot122)
  (clean shot474)
  (empty shaker113)
  (empty shot244)
  (empty shot122)
  (empty shot474)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker113 l0)
  (shaker-level shaker113 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail218 ingredient289)
  (cocktail-part2 cocktail218 ingredient116)
)
 (:goal
  (and
      (contains shot244 cocktail218)
      (contains shot122 cocktail218)
)))
