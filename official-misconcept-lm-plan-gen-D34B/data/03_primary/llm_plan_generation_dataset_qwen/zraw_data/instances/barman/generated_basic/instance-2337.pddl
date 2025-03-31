(define (problem prob)
 (:domain barman)
 (:objects 
      shaker467 - shaker
      left right - hand
      shot221 - shot
      ingredient132 ingredient245 - ingredient
      cocktail161 - cocktail
      dispenser454 dispenser492 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker467)
  (ontable shot221)
  (dispenses dispenser454 ingredient132)
  (dispenses dispenser492 ingredient245)
  (clean shaker467)
  (clean shot221)
  (empty shaker467)
  (empty shot221)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker467 l0)
  (shaker-level shaker467 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail161 ingredient245)
  (cocktail-part2 cocktail161 ingredient132)
)
 (:goal
  (and
      (contains shot221 cocktail161)
)))
