(define (problem prob)
 (:domain barman)
 (:objects 
      shaker409 - shaker
      left right - hand
      shot35 shot278 - shot
      ingredient436 ingredient343 ingredient493 ingredient13 - ingredient
      cocktail486 - cocktail
      dispenser449 dispenser447 dispenser467 dispenser437 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker409)
  (ontable shot35)
  (ontable shot278)
  (dispenses dispenser449 ingredient436)
  (dispenses dispenser447 ingredient343)
  (dispenses dispenser467 ingredient493)
  (dispenses dispenser437 ingredient13)
  (clean shaker409)
  (clean shot35)
  (clean shot278)
  (empty shaker409)
  (empty shot35)
  (empty shot278)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker409 l0)
  (shaker-level shaker409 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail486 ingredient343)
  (cocktail-part2 cocktail486 ingredient493)
)
 (:goal
  (and
      (contains shot35 cocktail486)
)))
