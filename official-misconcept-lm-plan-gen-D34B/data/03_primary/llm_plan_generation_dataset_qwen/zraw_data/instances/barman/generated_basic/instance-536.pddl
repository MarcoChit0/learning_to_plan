(define (problem prob)
 (:domain barman)
 (:objects 
      shaker352 - shaker
      left right - hand
      shot449 - shot
      ingredient381 ingredient23 ingredient87 - ingredient
      cocktail1 - cocktail
      dispenser239 dispenser142 dispenser67 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker352)
  (ontable shot449)
  (dispenses dispenser239 ingredient381)
  (dispenses dispenser142 ingredient23)
  (dispenses dispenser67 ingredient87)
  (clean shaker352)
  (clean shot449)
  (empty shaker352)
  (empty shot449)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker352 l0)
  (shaker-level shaker352 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail1 ingredient87)
  (cocktail-part2 cocktail1 ingredient23)
)
 (:goal
  (and
      (contains shot449 cocktail1)
)))
