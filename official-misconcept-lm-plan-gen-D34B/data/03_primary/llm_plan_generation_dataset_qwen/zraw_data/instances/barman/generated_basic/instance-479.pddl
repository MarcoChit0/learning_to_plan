(define (problem prob)
 (:domain barman)
 (:objects 
      shaker429 - shaker
      left right - hand
      shot366 shot231 - shot
      ingredient341 ingredient463 ingredient225 ingredient452 - ingredient
      cocktail1 - cocktail
      dispenser498 dispenser269 dispenser1 dispenser475 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker429)
  (ontable shot366)
  (ontable shot231)
  (dispenses dispenser498 ingredient341)
  (dispenses dispenser269 ingredient463)
  (dispenses dispenser1 ingredient225)
  (dispenses dispenser475 ingredient452)
  (clean shaker429)
  (clean shot366)
  (clean shot231)
  (empty shaker429)
  (empty shot366)
  (empty shot231)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker429 l0)
  (shaker-level shaker429 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail1 ingredient452)
  (cocktail-part2 cocktail1 ingredient341)
)
 (:goal
  (and
      (contains shot366 cocktail1)
)))
