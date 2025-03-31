(define (problem prob)
 (:domain barman)
 (:objects 
      shaker413 - shaker
      left right - hand
      shot228 shot241 - shot
      ingredient117 ingredient265 - ingredient
      cocktail1 - cocktail
      dispenser274 dispenser364 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker413)
  (ontable shot228)
  (ontable shot241)
  (dispenses dispenser274 ingredient117)
  (dispenses dispenser364 ingredient265)
  (clean shaker413)
  (clean shot228)
  (clean shot241)
  (empty shaker413)
  (empty shot228)
  (empty shot241)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker413 l0)
  (shaker-level shaker413 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail1 ingredient117)
  (cocktail-part2 cocktail1 ingredient265)
)
 (:goal
  (and
      (contains shot228 cocktail1)
)))
