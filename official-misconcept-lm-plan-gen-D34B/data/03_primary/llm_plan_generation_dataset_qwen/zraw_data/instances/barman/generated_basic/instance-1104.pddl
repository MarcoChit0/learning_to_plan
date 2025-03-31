(define (problem prob)
 (:domain barman)
 (:objects 
      shaker8 - shaker
      left right - hand
      shot448 shot441 - shot
      ingredient175 ingredient369 - ingredient
      cocktail1 - cocktail
      dispenser281 dispenser460 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker8)
  (ontable shot448)
  (ontable shot441)
  (dispenses dispenser281 ingredient175)
  (dispenses dispenser460 ingredient369)
  (clean shaker8)
  (clean shot448)
  (clean shot441)
  (empty shaker8)
  (empty shot448)
  (empty shot441)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker8 l0)
  (shaker-level shaker8 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail1 ingredient369)
  (cocktail-part2 cocktail1 ingredient175)
)
 (:goal
  (and
      (contains shot448 cocktail1)
)))
