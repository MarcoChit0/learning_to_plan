(define (problem prob)
 (:domain barman)
 (:objects 
      shaker440 - shaker
      left right - hand
      shot193 shot175 - shot
      ingredient392 ingredient443 - ingredient
      cocktail12 - cocktail
      dispenser334 dispenser272 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker440)
  (ontable shot193)
  (ontable shot175)
  (dispenses dispenser334 ingredient392)
  (dispenses dispenser272 ingredient443)
  (clean shaker440)
  (clean shot193)
  (clean shot175)
  (empty shaker440)
  (empty shot193)
  (empty shot175)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker440 l0)
  (shaker-level shaker440 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail12 ingredient392)
  (cocktail-part2 cocktail12 ingredient443)
)
 (:goal
  (and
      (contains shot193 cocktail12)
)))
