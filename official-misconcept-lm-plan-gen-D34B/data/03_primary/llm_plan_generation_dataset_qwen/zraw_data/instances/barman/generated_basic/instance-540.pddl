(define (problem prob)
 (:domain barman)
 (:objects 
      shaker113 - shaker
      left right - hand
      shot435 - shot
      ingredient222 ingredient265 ingredient443 - ingredient
      cocktail1 - cocktail
      dispenser369 dispenser405 dispenser304 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker113)
  (ontable shot435)
  (dispenses dispenser369 ingredient222)
  (dispenses dispenser405 ingredient265)
  (dispenses dispenser304 ingredient443)
  (clean shaker113)
  (clean shot435)
  (empty shaker113)
  (empty shot435)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker113 l0)
  (shaker-level shaker113 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail1 ingredient222)
  (cocktail-part2 cocktail1 ingredient265)
)
 (:goal
  (and
      (contains shot435 cocktail1)
)))
