(define (problem prob)
 (:domain barman)
 (:objects 
      shaker442 - shaker
      left right - hand
      shot178 shot414 - shot
      ingredient230 ingredient20 ingredient316 ingredient186 - ingredient
      cocktail33 - cocktail
      dispenser114 dispenser63 dispenser167 dispenser458 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker442)
  (ontable shot178)
  (ontable shot414)
  (dispenses dispenser114 ingredient230)
  (dispenses dispenser63 ingredient20)
  (dispenses dispenser167 ingredient316)
  (dispenses dispenser458 ingredient186)
  (clean shaker442)
  (clean shot178)
  (clean shot414)
  (empty shaker442)
  (empty shot178)
  (empty shot414)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker442 l0)
  (shaker-level shaker442 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail33 ingredient186)
  (cocktail-part2 cocktail33 ingredient230)
)
 (:goal
  (and
      (contains shot178 cocktail33)
)))
