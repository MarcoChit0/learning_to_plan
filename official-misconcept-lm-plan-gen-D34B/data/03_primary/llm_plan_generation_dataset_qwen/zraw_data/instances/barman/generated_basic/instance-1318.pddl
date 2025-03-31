(define (problem prob)
 (:domain barman)
 (:objects 
      shaker418 - shaker
      left right - hand
      shot316 - shot
      ingredient186 ingredient298 ingredient443 ingredient451 - ingredient
      cocktail453 - cocktail
      dispenser268 dispenser251 dispenser25 dispenser416 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker418)
  (ontable shot316)
  (dispenses dispenser268 ingredient186)
  (dispenses dispenser251 ingredient298)
  (dispenses dispenser25 ingredient443)
  (dispenses dispenser416 ingredient451)
  (clean shaker418)
  (clean shot316)
  (empty shaker418)
  (empty shot316)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker418 l0)
  (shaker-level shaker418 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail453 ingredient451)
  (cocktail-part2 cocktail453 ingredient186)
)
 (:goal
  (and
      (contains shot316 cocktail453)
)))
