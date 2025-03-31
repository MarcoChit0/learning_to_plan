(define (problem prob)
 (:domain barman)
 (:objects 
      shaker368 - shaker
      left right - hand
      shot464 shot474 shot165 - shot
      ingredient303 ingredient354 - ingredient
      cocktail1 cocktail2 - cocktail
      dispenser372 dispenser236 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker368)
  (ontable shot464)
  (ontable shot474)
  (ontable shot165)
  (dispenses dispenser372 ingredient303)
  (dispenses dispenser236 ingredient354)
  (clean shaker368)
  (clean shot464)
  (clean shot474)
  (clean shot165)
  (empty shaker368)
  (empty shot464)
  (empty shot474)
  (empty shot165)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker368 l0)
  (shaker-level shaker368 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail1 ingredient303)
  (cocktail-part2 cocktail1 ingredient354)
  (cocktail-part1 cocktail2 ingredient303)
  (cocktail-part2 cocktail2 ingredient354)
)
 (:goal
  (and
      (contains shot464 cocktail1)
      (contains shot474 cocktail2)
)))
