(define (problem prob)
 (:domain barman)
 (:objects 
      shaker238 - shaker
      left right - hand
      shot58 shot132 - shot
      ingredient209 ingredient175 ingredient186 ingredient486 - ingredient
      cocktail212 - cocktail
      dispenser144 dispenser103 dispenser143 dispenser263 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker238)
  (ontable shot58)
  (ontable shot132)
  (dispenses dispenser144 ingredient209)
  (dispenses dispenser103 ingredient175)
  (dispenses dispenser143 ingredient186)
  (dispenses dispenser263 ingredient486)
  (clean shaker238)
  (clean shot58)
  (clean shot132)
  (empty shaker238)
  (empty shot58)
  (empty shot132)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker238 l0)
  (shaker-level shaker238 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail212 ingredient186)
  (cocktail-part2 cocktail212 ingredient486)
)
 (:goal
  (and
      (contains shot58 cocktail212)
)))
