(define (problem prob)
 (:domain barman)
 (:objects 
      shaker356 - shaker
      left right - hand
      shot454 - shot
      ingredient435 ingredient398 ingredient36 - ingredient
      cocktail1 - cocktail
      dispenser0 dispenser105 dispenser467 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker356)
  (ontable shot454)
  (dispenses dispenser0 ingredient435)
  (dispenses dispenser105 ingredient398)
  (dispenses dispenser467 ingredient36)
  (clean shaker356)
  (clean shot454)
  (empty shaker356)
  (empty shot454)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker356 l0)
  (shaker-level shaker356 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail1 ingredient435)
  (cocktail-part2 cocktail1 ingredient36)
)
 (:goal
  (and
      (contains shot454 cocktail1)
)))
