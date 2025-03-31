(define (problem prob)
 (:domain barman)
 (:objects 
      shaker49 - shaker
      left right - hand
      shot487 shot456 - shot
      ingredient368 ingredient87 ingredient465 - ingredient
      cocktail1 - cocktail
      dispenser42 dispenser107 dispenser273 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker49)
  (ontable shot487)
  (ontable shot456)
  (dispenses dispenser42 ingredient368)
  (dispenses dispenser107 ingredient87)
  (dispenses dispenser273 ingredient465)
  (clean shaker49)
  (clean shot487)
  (clean shot456)
  (empty shaker49)
  (empty shot487)
  (empty shot456)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker49 l0)
  (shaker-level shaker49 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail1 ingredient465)
  (cocktail-part2 cocktail1 ingredient87)
)
 (:goal
  (and
      (contains shot487 cocktail1)
)))
