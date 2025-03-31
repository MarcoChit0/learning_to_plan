(define (problem prob)
 (:domain barman)
 (:objects 
      shaker7 - shaker
      left right - hand
      shot4 shot343 - shot
      ingredient137 ingredient289 - ingredient
      cocktail1 - cocktail
      dispenser303 dispenser402 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker7)
  (ontable shot4)
  (ontable shot343)
  (dispenses dispenser303 ingredient137)
  (dispenses dispenser402 ingredient289)
  (clean shaker7)
  (clean shot4)
  (clean shot343)
  (empty shaker7)
  (empty shot4)
  (empty shot343)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker7 l0)
  (shaker-level shaker7 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail1 ingredient289)
  (cocktail-part2 cocktail1 ingredient137)
)
 (:goal
  (and
      (contains shot4 cocktail1)
)))
