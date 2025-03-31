(define (problem prob)
 (:domain barman)
 (:objects 
      shaker57 - shaker
      left right - hand
      shot115 - shot
      ingredient313 ingredient147 ingredient471 ingredient174 - ingredient
      cocktail1 - cocktail
      dispenser156 dispenser146 dispenser424 dispenser84 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker57)
  (ontable shot115)
  (dispenses dispenser156 ingredient313)
  (dispenses dispenser146 ingredient147)
  (dispenses dispenser424 ingredient471)
  (dispenses dispenser84 ingredient174)
  (clean shaker57)
  (clean shot115)
  (empty shaker57)
  (empty shot115)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker57 l0)
  (shaker-level shaker57 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail1 ingredient313)
  (cocktail-part2 cocktail1 ingredient471)
)
 (:goal
  (and
      (contains shot115 cocktail1)
)))
