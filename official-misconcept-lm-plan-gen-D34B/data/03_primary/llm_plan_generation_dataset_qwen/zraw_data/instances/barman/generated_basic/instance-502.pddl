(define (problem prob)
 (:domain barman)
 (:objects 
      shaker416 - shaker
      left right - hand
      shot451 shot11 - shot
      ingredient266 ingredient399 ingredient97 ingredient373 - ingredient
      cocktail1 - cocktail
      dispenser268 dispenser302 dispenser13 dispenser175 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker416)
  (ontable shot451)
  (ontable shot11)
  (dispenses dispenser268 ingredient266)
  (dispenses dispenser302 ingredient399)
  (dispenses dispenser13 ingredient97)
  (dispenses dispenser175 ingredient373)
  (clean shaker416)
  (clean shot451)
  (clean shot11)
  (empty shaker416)
  (empty shot451)
  (empty shot11)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker416 l0)
  (shaker-level shaker416 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail1 ingredient373)
  (cocktail-part2 cocktail1 ingredient399)
)
 (:goal
  (and
      (contains shot451 cocktail1)
)))
