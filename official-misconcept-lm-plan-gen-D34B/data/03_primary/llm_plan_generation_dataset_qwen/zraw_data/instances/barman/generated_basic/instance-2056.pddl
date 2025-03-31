(define (problem prob)
 (:domain barman)
 (:objects 
      shaker420 - shaker
      left right - hand
      shot75 - shot
      ingredient221 ingredient119 ingredient140 ingredient149 - ingredient
      cocktail277 - cocktail
      dispenser75 dispenser437 dispenser381 dispenser0 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker420)
  (ontable shot75)
  (dispenses dispenser75 ingredient221)
  (dispenses dispenser437 ingredient119)
  (dispenses dispenser381 ingredient140)
  (dispenses dispenser0 ingredient149)
  (clean shaker420)
  (clean shot75)
  (empty shaker420)
  (empty shot75)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker420 l0)
  (shaker-level shaker420 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail277 ingredient149)
  (cocktail-part2 cocktail277 ingredient119)
)
 (:goal
  (and
      (contains shot75 cocktail277)
)))
