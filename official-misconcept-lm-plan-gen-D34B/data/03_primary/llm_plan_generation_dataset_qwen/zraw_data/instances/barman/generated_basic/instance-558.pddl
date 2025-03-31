(define (problem prob)
 (:domain barman)
 (:objects 
      shaker164 - shaker
      left right - hand
      shot412 shot40 - shot
      ingredient282 ingredient364 ingredient314 - ingredient
      cocktail1 - cocktail
      dispenser101 dispenser151 dispenser354 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker164)
  (ontable shot412)
  (ontable shot40)
  (dispenses dispenser101 ingredient282)
  (dispenses dispenser151 ingredient364)
  (dispenses dispenser354 ingredient314)
  (clean shaker164)
  (clean shot412)
  (clean shot40)
  (empty shaker164)
  (empty shot412)
  (empty shot40)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker164 l0)
  (shaker-level shaker164 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail1 ingredient314)
  (cocktail-part2 cocktail1 ingredient364)
)
 (:goal
  (and
      (contains shot412 cocktail1)
)))
