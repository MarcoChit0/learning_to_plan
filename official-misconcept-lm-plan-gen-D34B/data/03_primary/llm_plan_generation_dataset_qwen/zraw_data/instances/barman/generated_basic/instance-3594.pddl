(define (problem prob)
 (:domain barman)
 (:objects 
      shaker6 - shaker
      left right - hand
      shot218 shot268 - shot
      ingredient196 ingredient172 ingredient88 ingredient185 - ingredient
      cocktail68 - cocktail
      dispenser114 dispenser45 dispenser233 dispenser142 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker6)
  (ontable shot218)
  (ontable shot268)
  (dispenses dispenser114 ingredient196)
  (dispenses dispenser45 ingredient172)
  (dispenses dispenser233 ingredient88)
  (dispenses dispenser142 ingredient185)
  (clean shaker6)
  (clean shot218)
  (clean shot268)
  (empty shaker6)
  (empty shot218)
  (empty shot268)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker6 l0)
  (shaker-level shaker6 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail68 ingredient88)
  (cocktail-part2 cocktail68 ingredient172)
)
 (:goal
  (and
      (contains shot218 cocktail68)
)))
