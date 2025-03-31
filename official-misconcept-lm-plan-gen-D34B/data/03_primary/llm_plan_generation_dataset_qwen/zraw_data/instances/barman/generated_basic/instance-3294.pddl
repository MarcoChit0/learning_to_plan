(define (problem prob)
 (:domain barman)
 (:objects 
      shaker255 - shaker
      left right - hand
      shot379 - shot
      ingredient242 ingredient220 - ingredient
      cocktail2 - cocktail
      dispenser135 dispenser455 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker255)
  (ontable shot379)
  (dispenses dispenser135 ingredient242)
  (dispenses dispenser455 ingredient220)
  (clean shaker255)
  (clean shot379)
  (empty shaker255)
  (empty shot379)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker255 l0)
  (shaker-level shaker255 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail2 ingredient220)
  (cocktail-part2 cocktail2 ingredient242)
)
 (:goal
  (and
      (contains shot379 cocktail2)
)))
