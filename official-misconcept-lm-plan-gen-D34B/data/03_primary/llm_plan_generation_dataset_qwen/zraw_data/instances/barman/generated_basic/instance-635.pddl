(define (problem prob)
 (:domain barman)
 (:objects 
      shaker301 - shaker
      left right - hand
      shot412 shot303 shot334 - shot
      ingredient178 ingredient366 ingredient400 - ingredient
      cocktail1 - cocktail
      dispenser212 dispenser136 dispenser183 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker301)
  (ontable shot412)
  (ontable shot303)
  (ontable shot334)
  (dispenses dispenser212 ingredient178)
  (dispenses dispenser136 ingredient366)
  (dispenses dispenser183 ingredient400)
  (clean shaker301)
  (clean shot412)
  (clean shot303)
  (clean shot334)
  (empty shaker301)
  (empty shot412)
  (empty shot303)
  (empty shot334)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker301 l0)
  (shaker-level shaker301 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail1 ingredient366)
  (cocktail-part2 cocktail1 ingredient400)
)
 (:goal
  (and
      (contains shot412 cocktail1)
      (contains shot303 ingredient366)
)))
