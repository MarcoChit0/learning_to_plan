(define (problem prob)
 (:domain barman)
 (:objects 
      shaker53 - shaker
      left right - hand
      shot451 - shot
      ingredient12 ingredient58 ingredient5 ingredient183 - ingredient
      cocktail1 - cocktail
      dispenser437 dispenser313 dispenser244 dispenser372 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker53)
  (ontable shot451)
  (dispenses dispenser437 ingredient12)
  (dispenses dispenser313 ingredient58)
  (dispenses dispenser244 ingredient5)
  (dispenses dispenser372 ingredient183)
  (clean shaker53)
  (clean shot451)
  (empty shaker53)
  (empty shot451)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker53 l0)
  (shaker-level shaker53 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail1 ingredient5)
  (cocktail-part2 cocktail1 ingredient12)
)
 (:goal
  (and
      (contains shot451 cocktail1)
)))
