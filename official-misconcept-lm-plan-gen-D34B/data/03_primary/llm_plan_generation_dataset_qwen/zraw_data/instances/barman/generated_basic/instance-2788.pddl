(define (problem prob)
 (:domain barman)
 (:objects 
      shaker179 - shaker
      left right - hand
      shot187 - shot
      ingredient51 ingredient70 ingredient59 ingredient265 - ingredient
      cocktail206 - cocktail
      dispenser292 dispenser423 dispenser171 dispenser207 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker179)
  (ontable shot187)
  (dispenses dispenser292 ingredient51)
  (dispenses dispenser423 ingredient70)
  (dispenses dispenser171 ingredient59)
  (dispenses dispenser207 ingredient265)
  (clean shaker179)
  (clean shot187)
  (empty shaker179)
  (empty shot187)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker179 l0)
  (shaker-level shaker179 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail206 ingredient59)
  (cocktail-part2 cocktail206 ingredient70)
)
 (:goal
  (and
      (contains shot187 cocktail206)
)))
