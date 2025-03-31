(define (problem prob)
 (:domain barman)
 (:objects 
      shaker103 - shaker
      left right - hand
      shot329 shot31 - shot
      ingredient162 ingredient170 ingredient361 ingredient62 - ingredient
      cocktail336 - cocktail
      dispenser409 dispenser29 dispenser472 dispenser62 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker103)
  (ontable shot329)
  (ontable shot31)
  (dispenses dispenser409 ingredient162)
  (dispenses dispenser29 ingredient170)
  (dispenses dispenser472 ingredient361)
  (dispenses dispenser62 ingredient62)
  (clean shaker103)
  (clean shot329)
  (clean shot31)
  (empty shaker103)
  (empty shot329)
  (empty shot31)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker103 l0)
  (shaker-level shaker103 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail336 ingredient162)
  (cocktail-part2 cocktail336 ingredient62)
)
 (:goal
  (and
      (contains shot329 cocktail336)
)))
