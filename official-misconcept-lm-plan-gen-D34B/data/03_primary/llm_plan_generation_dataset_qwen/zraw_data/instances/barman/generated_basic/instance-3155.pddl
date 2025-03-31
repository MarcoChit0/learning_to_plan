(define (problem prob)
 (:domain barman)
 (:objects 
      shaker86 - shaker
      left right - hand
      shot260 shot277 shot22 - shot
      ingredient218 ingredient127 ingredient142 ingredient437 - ingredient
      cocktail16 - cocktail
      dispenser238 dispenser437 dispenser164 dispenser170 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker86)
  (ontable shot260)
  (ontable shot277)
  (ontable shot22)
  (dispenses dispenser238 ingredient218)
  (dispenses dispenser437 ingredient127)
  (dispenses dispenser164 ingredient142)
  (dispenses dispenser170 ingredient437)
  (clean shaker86)
  (clean shot260)
  (clean shot277)
  (clean shot22)
  (empty shaker86)
  (empty shot260)
  (empty shot277)
  (empty shot22)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker86 l0)
  (shaker-level shaker86 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail16 ingredient218)
  (cocktail-part2 cocktail16 ingredient127)
)
 (:goal
  (and
      (contains shot260 cocktail16)
      (contains shot277 cocktail16)
)))
