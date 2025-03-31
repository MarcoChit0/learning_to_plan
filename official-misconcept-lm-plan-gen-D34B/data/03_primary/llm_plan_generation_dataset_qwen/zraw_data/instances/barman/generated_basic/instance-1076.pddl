(define (problem prob)
 (:domain barman)
 (:objects 
      shaker26 - shaker
      left right - hand
      shot227 shot489 shot401 - shot
      ingredient329 ingredient152 ingredient361 ingredient112 - ingredient
      cocktail1 - cocktail
      dispenser298 dispenser213 dispenser381 dispenser1 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker26)
  (ontable shot227)
  (ontable shot489)
  (ontable shot401)
  (dispenses dispenser298 ingredient329)
  (dispenses dispenser213 ingredient152)
  (dispenses dispenser381 ingredient361)
  (dispenses dispenser1 ingredient112)
  (clean shaker26)
  (clean shot227)
  (clean shot489)
  (clean shot401)
  (empty shaker26)
  (empty shot227)
  (empty shot489)
  (empty shot401)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker26 l0)
  (shaker-level shaker26 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail1 ingredient361)
  (cocktail-part2 cocktail1 ingredient112)
)
 (:goal
  (and
      (contains shot227 cocktail1)
      (contains shot489 cocktail1)
)))
