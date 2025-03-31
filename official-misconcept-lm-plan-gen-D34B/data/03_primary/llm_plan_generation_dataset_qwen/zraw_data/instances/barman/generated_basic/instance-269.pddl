(define (problem prob)
 (:domain barman)
 (:objects 
      shaker134 - shaker
      left right - hand
      shot16 shot462 - shot
      ingredient130 ingredient84 ingredient355 - ingredient
      cocktail1 - cocktail
      dispenser463 dispenser218 dispenser437 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker134)
  (ontable shot16)
  (ontable shot462)
  (dispenses dispenser463 ingredient130)
  (dispenses dispenser218 ingredient84)
  (dispenses dispenser437 ingredient355)
  (clean shaker134)
  (clean shot16)
  (clean shot462)
  (empty shaker134)
  (empty shot16)
  (empty shot462)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker134 l0)
  (shaker-level shaker134 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail1 ingredient84)
  (cocktail-part2 cocktail1 ingredient355)
)
 (:goal
  (and
      (contains shot16 cocktail1)
)))
