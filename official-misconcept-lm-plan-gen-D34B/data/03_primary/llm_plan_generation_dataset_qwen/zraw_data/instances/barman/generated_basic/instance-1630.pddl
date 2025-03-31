(define (problem prob)
 (:domain barman)
 (:objects 
      shaker379 - shaker
      left right - hand
      shot40 shot191 shot376 - shot
      ingredient310 ingredient170 - ingredient
      cocktail307 - cocktail
      dispenser455 dispenser71 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker379)
  (ontable shot40)
  (ontable shot191)
  (ontable shot376)
  (dispenses dispenser455 ingredient310)
  (dispenses dispenser71 ingredient170)
  (clean shaker379)
  (clean shot40)
  (clean shot191)
  (clean shot376)
  (empty shaker379)
  (empty shot40)
  (empty shot191)
  (empty shot376)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker379 l0)
  (shaker-level shaker379 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail307 ingredient310)
  (cocktail-part2 cocktail307 ingredient170)
)
 (:goal
  (and
      (contains shot40 cocktail307)
      (contains shot191 cocktail307)
)))
