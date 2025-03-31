(define (problem prob)
 (:domain barman)
 (:objects 
      shaker422 - shaker
      left right - hand
      shot383 - shot
      ingredient379 ingredient485 ingredient161 ingredient229 - ingredient
      cocktail1 - cocktail
      dispenser67 dispenser307 dispenser43 dispenser148 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker422)
  (ontable shot383)
  (dispenses dispenser67 ingredient379)
  (dispenses dispenser307 ingredient485)
  (dispenses dispenser43 ingredient161)
  (dispenses dispenser148 ingredient229)
  (clean shaker422)
  (clean shot383)
  (empty shaker422)
  (empty shot383)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker422 l0)
  (shaker-level shaker422 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail1 ingredient485)
  (cocktail-part2 cocktail1 ingredient229)
)
 (:goal
  (and
      (contains shot383 cocktail1)
)))
