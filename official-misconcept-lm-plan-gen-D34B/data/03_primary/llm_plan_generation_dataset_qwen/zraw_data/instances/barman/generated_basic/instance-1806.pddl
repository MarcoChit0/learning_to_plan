(define (problem prob)
 (:domain barman)
 (:objects 
      shaker247 - shaker
      left right - hand
      shot404 shot181 - shot
      ingredient147 ingredient197 ingredient388 ingredient164 - ingredient
      cocktail359 - cocktail
      dispenser418 dispenser474 dispenser209 dispenser484 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker247)
  (ontable shot404)
  (ontable shot181)
  (dispenses dispenser418 ingredient147)
  (dispenses dispenser474 ingredient197)
  (dispenses dispenser209 ingredient388)
  (dispenses dispenser484 ingredient164)
  (clean shaker247)
  (clean shot404)
  (clean shot181)
  (empty shaker247)
  (empty shot404)
  (empty shot181)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker247 l0)
  (shaker-level shaker247 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail359 ingredient197)
  (cocktail-part2 cocktail359 ingredient388)
)
 (:goal
  (and
      (contains shot404 cocktail359)
)))
