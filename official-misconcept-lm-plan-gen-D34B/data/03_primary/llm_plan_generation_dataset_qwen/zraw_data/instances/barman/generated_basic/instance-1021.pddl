(define (problem prob)
 (:domain barman)
 (:objects 
      shaker467 - shaker
      left right - hand
      shot68 - shot
      ingredient296 ingredient146 ingredient147 - ingredient
      cocktail1 - cocktail
      dispenser250 dispenser313 dispenser62 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker467)
  (ontable shot68)
  (dispenses dispenser250 ingredient296)
  (dispenses dispenser313 ingredient146)
  (dispenses dispenser62 ingredient147)
  (clean shaker467)
  (clean shot68)
  (empty shaker467)
  (empty shot68)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker467 l0)
  (shaker-level shaker467 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail1 ingredient147)
  (cocktail-part2 cocktail1 ingredient146)
)
 (:goal
  (and
      (contains shot68 cocktail1)
)))
