(define (problem prob)
 (:domain barman)
 (:objects 
      shaker195 - shaker
      left right - hand
      shot378 - shot
      ingredient126 ingredient389 ingredient170 ingredient161 - ingredient
      cocktail385 - cocktail
      dispenser312 dispenser181 dispenser427 dispenser55 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker195)
  (ontable shot378)
  (dispenses dispenser312 ingredient126)
  (dispenses dispenser181 ingredient389)
  (dispenses dispenser427 ingredient170)
  (dispenses dispenser55 ingredient161)
  (clean shaker195)
  (clean shot378)
  (empty shaker195)
  (empty shot378)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker195 l0)
  (shaker-level shaker195 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail385 ingredient170)
  (cocktail-part2 cocktail385 ingredient126)
)
 (:goal
  (and
      (contains shot378 cocktail385)
)))
