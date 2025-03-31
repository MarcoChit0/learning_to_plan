(define (problem prob)
 (:domain barman)
 (:objects 
      shaker177 - shaker
      left right - hand
      shot162 shot48 - shot
      ingredient226 ingredient278 - ingredient
      cocktail311 - cocktail
      dispenser386 dispenser486 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker177)
  (ontable shot162)
  (ontable shot48)
  (dispenses dispenser386 ingredient226)
  (dispenses dispenser486 ingredient278)
  (clean shaker177)
  (clean shot162)
  (clean shot48)
  (empty shaker177)
  (empty shot162)
  (empty shot48)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker177 l0)
  (shaker-level shaker177 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail311 ingredient278)
  (cocktail-part2 cocktail311 ingredient226)
)
 (:goal
  (and
      (contains shot162 cocktail311)
)))
