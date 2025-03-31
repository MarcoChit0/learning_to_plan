(define (problem prob)
 (:domain barman)
 (:objects 
      shaker331 - shaker
      left right - hand
      shot251 - shot
      ingredient346 ingredient340 - ingredient
      cocktail206 - cocktail
      dispenser373 dispenser262 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker331)
  (ontable shot251)
  (dispenses dispenser373 ingredient346)
  (dispenses dispenser262 ingredient340)
  (clean shaker331)
  (clean shot251)
  (empty shaker331)
  (empty shot251)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker331 l0)
  (shaker-level shaker331 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail206 ingredient346)
  (cocktail-part2 cocktail206 ingredient340)
)
 (:goal
  (and
      (contains shot251 cocktail206)
)))
