(define (problem prob)
 (:domain barman)
 (:objects 
      shaker116 - shaker
      left right - hand
      shot481 shot58 shot268 - shot
      ingredient200 ingredient229 ingredient346 ingredient410 - ingredient
      cocktail33 - cocktail
      dispenser303 dispenser458 dispenser396 dispenser116 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker116)
  (ontable shot481)
  (ontable shot58)
  (ontable shot268)
  (dispenses dispenser303 ingredient200)
  (dispenses dispenser458 ingredient229)
  (dispenses dispenser396 ingredient346)
  (dispenses dispenser116 ingredient410)
  (clean shaker116)
  (clean shot481)
  (clean shot58)
  (clean shot268)
  (empty shaker116)
  (empty shot481)
  (empty shot58)
  (empty shot268)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker116 l0)
  (shaker-level shaker116 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail33 ingredient346)
  (cocktail-part2 cocktail33 ingredient229)
)
 (:goal
  (and
      (contains shot481 cocktail33)
      (contains shot58 cocktail33)
)))
